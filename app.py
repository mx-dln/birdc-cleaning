from flask import Flask, request, jsonify, send_file, render_template, Response
import pandas as pd
import os
import matplotlib.pyplot as plt
import chardet
import re
import unicodedata
from io import BytesIO, StringIO
import html
from dateutil import parser
from word2number import w2n
from rapidfuzz import fuzz, process
import string
import io
from difflib import SequenceMatcher


app = Flask(__name__)
uploaded_df = pd.DataFrame()
history_stack = []

def normalize_unicode(df):
    return df.applymap(lambda x: unicodedata.normalize('NFKC', str(x)) if isinstance(x, str) else x)

def save_history():
    global uploaded_df, history_stack
    history_stack.append(uploaded_df.copy())

def dataframe_to_html_with_id(df):
    html = "<table class='dataframe table table-striped'>"
    
    # HEADER
    html += "<thead><tr>"
    html += "<th style='width: 50px;'>Edit</th>"
    for col in df.columns:
        html += f"<th>{col}</th>"
    html += "</tr></thead><tbody>"

    # ROWS WITH data-id
    for i, row in df.iterrows():
        html += f"<tr data-id='{i}'>"
        html += "<td><button class='btn btn-sm btn-outline-primary edit-btn' onclick='editRow(this)' data-row-id='" + str(i) + "'><i class='bi bi-pencil'></i></button></td>"
        for val in row:
            html += f"<td>{val}</td>"
        html += "</tr>"

    html += "</tbody></table>"
    return html

# ---------------------------
# DTI Mapping Functions
# ---------------------------
def normalize_text(text):
    # Handle NaN/float values by converting to empty string
    if pd.isna(text) or text is None:
        return ""
    
    s = str(text)  # Convert to string first
    # unify separators, remove excessive punctuation, keep words
    s = s.replace("–", "-").replace("—", "-").replace("&", " and ")
    s = re.sub(r"[()\"'<>;:]", " ", s)
    s = re.sub(r"[/\\_\[\]\{\}]", " ", s)
    s = re.sub(r"[^0-9A-Za-z\-_\.&, ]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()

# DTI categories (canonical strings)
DTI_CATEGORIES = {
    "Manufacturing Of Essential Goods": "Manufacturing Of Essential Goods",
    "Food And Beverages (Only Non-Alcoholic Drinks)": "Food And Beverages (Only Non-Alcoholic Drinks)",
    "Essential Retail": "Essential Retail (E.G., Groceries, Markets, Convenience Stores, Drug Stores)",
    "Food_Preparation_TakeOut": "Food Preparation Insofar As Take-Out And Delivery Services",
    "Water-Refilling Stores": "Water-Refilling Stores",
    "Laundry Services": "Laundry Services (Including Self-Service)",
    "Gasoline Stations": "Gasoline Stations",
    "Construction Projects": "Public And Private Construction Projects That Are Essential...",
    "Publishing And Printing": "Publishing And Printing Services...",
    "Veterinary Activities": "Veterinary Activities",
    "Hotel/Accommodation": "Hotel And Other Accommodation Establishments",
    "Hardware Stores": "Hardware Stores",
    "Clothing And Accessories": "Clothing And Accessories",
    "Bookstore/OfficeSupplies": "Bookstores And School And Office Supplies Stores",
    "Baby Supplies": "Baby Or Infant Care Supplies Stores",
    "Pet Shops": "Pet Shops, Pet Food And Pet Care Supplies",
    "IT/Computer": "It, Communications, And Electronic Equipment",
    "Flower/Jewelry": "Flower, Jewelry, Novelty, Antique, Perfume Shops",
    "Toy Store": "Toy Store (With Playgrounds And Amusement Area Closed)",
    "Music Stores": "Music Stores",
    "Art Galleries": "Art Galleries (Selling Only)",
    "Firearms": "Firearms And Ammunition Trading Establishment",
    "Salon/Barber": "Barbershops And Salons, Provided That They Are Compliant With Dti-Issued Protocols",
    "Gyms/Fitness": "Gyms/Fitness Studios And Sports Facilities",
    "Internet/ComputerShops": "Internet And Computer Shops (Subject To Strict Health Protocols) Only For Work Or Educational Purposes Only",
    "Funeral": "Funeral And Embalming Services",
    "AutoRepair": "Repair Of Motor Vehicles, Motorcycles, And Bicycles, (Including Vulcanizing Shops, Battery Repair Shops, Auto Repair Shops, Car Wash)",
    "Delivery/Logistics": "Delivery And Courier Services, Whether In-House Or Outsourced Transporting Food, Medicine, Or Other Essential Goods, Including Clothing, Accessories, Hardware, House Wares, School And Office Supplies, As Well As Pet Food And Other Veterinary Products",
    "RealEstate Leasing": "Real Estate Activities (Including Parking Space Leasing Companies) Leasing",
    "Dining Dine In": "Dining/ Restaurants Dine In",
    "Dining Take Out": "Dining/ Restaurants Delivery And Take-Out Only",
    "Office/Admin": "Office Administrative And Office Support (Such As Photocopying, Billing, And Record Keeping Services)",
    "Other Services": "Other Services Such As Photography Services Fashion, Industrial, Graphic, And Interior Design)"
}

DTI_CANONICAL = list(DTI_CATEGORIES.values())

# DTI mapping rules
RULE_DEFS = [
    # priority 1: Manufacturing (food processors)
    (r"\b(food processor|food products|food processing|meat processor|meat processing|frozen good|frozen goods|can[n]?ery|processing plant|packing plant|packer)\b", "Manufacturing Of Essential Goods", 1),

    # priority 2: Dining / Food preparation (take-out)
    (r"\b(burger stand|food stand|food stall|snack bar|snackbar|panciteria|kitchen|food services|refreshment|refreshments|refreshment stand|grill|grilled|bbq|barbecue)\b", "Food_Preparation_TakeOut", 2),

    # priority 3: Dining dine-in explicit
    (r"\b(restaurant|resto|eatery|kainan|dine in|dine-in|diner|vídeoke|videoke|karaoke.*bar|license to serve)\b", "Dining Dine In", 3),

    # priority 4: Essential retail specifics
    (r"\b(sari[- ]?sari|grocery|groceries|grocer|rice retailer|rice seller|rice|corn retailer|corn|dried fish|dried-fish|driedfish|vegetable vendor|veg vendor|meat vendor|fish vendor|market|market stall|market vendor|balot|balut|palay|corn trading)\b", "Essential Retail", 4),

    # priority 5: Drugstore/pharmacy
    (r"\b(drug store|pharmacy|botika|drugstore)\b", "Essential Retail", 5),

    # priority 6: Water refill
    (r"\b(water refill|water refilling|refilling station|water filling|water station)\b", "Water-Refilling Stores", 6),

    # priority 7: Gasoline / fuel
    (r"\b(gasoline|gas station|petrol station|fuel station|filling station)\b", "Gasoline Stations", 7),

    # priority 8: Construction projects & suppliers
    (r"\b(construction|construction services|contractor|construction supply|construction materials|aggregate|aggregates|cement|steel|builder|building contractor)\b", "Construction Projects", 8),

    # priority 9: Hardware / lumber / DIY
    (r"\b(hardware|lumber|hardware store|construction supply - retail|construction supply)\b", "Hardware Stores", 9),

    # priority 10: Auto repair / motor parts / vulcanizing
    (r"\b(vulcaniz|vulcanizing|vulcanize|auto repair|auto works|motor parts|auto supply|battery repair|repair shop|car wash|auto shop|mechanic)\b", "AutoRepair", 10),

    # priority 11: Computer / internet shops / IT equipment
    (r"\b(computer shop|internet cafe|internet shop|it and communication|computer repair|computer retailer|computer dealer)\b", "Internet/ComputerShops", 11),

    # priority 12: Veterinary / pet related
    (r"\b(veterinary|vet clinic|animal clinic|pet shop|pet food|veterinary clinic)\b", "Veterinary Activities", 12),

    # priority 13: Hotel / boarding house / accommodation
    (r"\b(hotel|motel|boarding house|apartelle|resort|lodging|accommodation)\b", "Hotel/Accommodation", 13),

    # priority 14: Gym / fitness
    (r"\b(gym|fitness|fitness studio|fitness center|sports facility|sports centre|sports center|gymnasium)\b", "Gyms/Fitness", 14),

    # priority 15: Salon / barber
    (r"\b(salon|barber|barbershop|parlor|parlour|beauty shop|beauty products)\b", "Salon/Barber", 15),

    # priority 16: Printing / publishing / office equipment
    (r"\b(print|printing|printing services|printing press|publisher|publishing)\b", "Publishing And Printing", 16),

    # priority 17: Bookstore / school & office supplies
    (r"\b(bookstore|school supplies|stationery|office supplies)\b", "Bookstore/OfficeSupplies", 17),

    # priority 18: Clothing / boutique / ukay
    (r"\b(boutique|clothes|clothing|apparel|ukay|garment|garments|tailor|dressmaker)\b", "Clothing And Accessories", 18),

    # priority 19: Toy / music / art
    (r"\b(toy store|toy\b|playground|amusement)\b", "Toy Store", 19),
    (r"\bmusic store\b|\bmusic\b", "Music Stores", 19),
    (r"\bart gallery\b|\bgallery\b", "Art Galleries", 19),

    # priority 20: Real estate / lessor / renting / leasing
    (r"\blessor\b|\bleasing\b|\brental\b|\brentals\b|\bapartment\b|\breal property lessor\b|\breal estate office\b", "RealEstate Leasing", 20),

    # priority 21: Funeral
    (r"\b(funeral|embalm|embalming)\b", "Funeral", 21),

    # priority 22: Delivery/logistics
    (r"\b(delivery|courier|logistics|freight|cargo|warehouse|warehousing|shipping|freight forward)\b", "Delivery/Logistics", 22),

    # priority 23: Generic "dealer/trading" fallback
    (r"\b(dealer|dealer of|dealer/|trading|trader|trading company|wholesaler|wholesale|retailer|retail)\b", None, 23),

    # priority 99: Everything else -> Other Services
    (r".+", "Other Services", 99),
]

# Token sets for special logic
ESSENTIAL_KEYWORDS = {
    "rice", "corn", "market", "grocery", "groceries", "sari-sari", "sari sari", "dried", "driedfish", "dried-fish",
    "vegetable", "veg", "meat", "fish", "balut", "balot", "palay", "bakery", "bakery goods", "bakery shop"
}
MANUFACTURING_KEYWORDS = {
    "food processor", "food processing", "processing", "packing", "packer", "meat processor", "frozen goods", "canning", "canner", "canning plant"
}
JUNK_KEYWORDS = {"junkshop", "junk shop", "scrap", "scrapyard", "waste dealer"}

# DTI Subcategory Violation Detection Rules
VIOLATION_RULES = {
    "Dining Dine In": [
        (r"\b(videoke|karaoke|bar|club|disco|pub|nightclub|live band|live music|entertainment)\b", "Prohibited entertainment services"),
        (r"\b(alcohol|liquor|beer|wine|spirits|cocktail)\b", "Alcohol service violation"),
        (r"\b(24 hours|24hr|24-hr|open 24|all night)\b", "Operating hours violation"),
    ],
    "Gyms/Fitness": [
        (r"\b(pool|swimming|spa|sauna|steam|massage)\b", "Non-fitness services offered"),
        (r"\b(group class|group fitness|dance class|yoga class|zumba)\b", "Group class violation"),
    ],
    "Internet/ComputerShops": [
        (r"\b(gaming|games|online games|computer games|dota|lol|mobile legends)\b", "Gaming/computer games violation"),
        (r"\b(social media|facebook|youtube|tiktok|instagram|twitter)\b", "Social media/entertainment use"),
    ],
    "Salon/Barber": [
        (r"\b(spa|massage|facial|wellness|therapy)\b", "Non-salon services violation"),
        (r"\b(nail salon|nail art|manicure|pedicure)\b", "Nail services may require separate permit"),
    ],
    "Hotel/Accommodation": [
        (r"\b(short time|hourly|motel|drive-in)\b", "Motel-style operations violation"),
        (r"\b(casino|gambling|betting)\b", "Gambling services violation"),
    ],
    "Construction Projects": [
        (r"\b(demolition|wrecking|destruction)\b", "Demolition without proper permit"),
        (r"\b(excavation|digging|trenching)\b", "Excavation work violation"),
    ],
    "AutoRepair": [
        (r"\b(modified|modification|custom|race|tuning)\b", "Vehicle modification violation"),
        (r"\b(paint|body work|dent|collision)\b", "Body work may require separate permit"),
    ],
    "Hardware Stores": [
        (r"\b(explosive|dynamite|fireworks|pyrotechnics)\b", "Explosives/firearms violation"),
        (r"\b(chemical|hazardous|toxic|corrosive)\b", "Hazardous materials violation"),
    ],
    "Essential Retail": [
        (r"\b(alcohol|liquor|beer|wine|spirits)\b", "Alcohol sales without proper permit"),
        (r"\b(tobacco|cigarette|vape|e-cigarette)\b", "Tobacco/vape sales violation"),
    ],
    "Food_Preparation_TakeOut": [
        (r"\b(dine in|eat in|sit down|table|chair)\b", "Dine-in services violation"),
        (r"\b(buffet|self service)\b", "Buffet service violation"),
    ],
    "Water-Refilling Stores": [
        (r"\b(mineral|spring|alkaline|ionized)\b", "False health claims violation"),
        (r"\b(delivery|home service)\b", "Delivery services may require separate permit"),
    ],
    "Laundry Services": [
        (r"\b(dry cleaning|dry clean)\b", "Dry cleaning requires separate permit"),
        (r"\b(industrial|commercial|heavy duty)\b", "Industrial laundry may require separate permit"),
    ],
    "Gasoline Stations": [
        (r"\b(repair|mechanic|auto service)\b", "Auto repair services violation"),
        (r"\b(convenience store|mini mart|sari sari)\b", "Retail services without proper permit"),
    ],
}

def best_difflib_match(text: str, cutoff: float = 0.60):
    """Return best matching canonical DTI category string and score, or (None, 0.0)."""
    t = (text or "").lower().strip()
    if not t:
        return None, 0.0
    best = None
    best_score = 0.0
    for cand in DTI_CANONICAL:
        score = SequenceMatcher(None, t, cand.lower()).ratio()
        if score > best_score:
            best_score = score
            best = cand
    if best_score >= cutoff:
        return best, best_score
    return None, best_score

def map_lob_to_dti(lob_text: str):
    text = (lob_text or "").strip().lower()
    if text == "":
        return DTI_CATEGORIES["Other Services"], "empty", 1.0

    t = text

    # If the whole row mentions JUNKSHOP -> Other Services
    if any(k in t for k in JUNK_KEYWORDS):
        return DTI_CATEGORIES["Other Services"], "rule_junkshop", 1.0

    # If any manufacturing keywords present -> manufacturing
    if any(k in t for k in MANUFACTURING_KEYWORDS):
        return DTI_CATEGORIES["Manufacturing Of Essential Goods"], "rule_manufacturing_keyword", 1.0

    # token-level evaluation: gather matches with priorities
    matches = []
    for pattern, cat_key, priority in RULE_DEFS:
        try:
            if re.search(pattern, t):
                matches.append((priority, cat_key, pattern))
        except re.error:
            continue

    # If we have concrete category matches, pick the one with lowest priority value
    concrete = [m for m in matches if m[1] is not None]
    if concrete:
        concrete.sort(key=lambda x: x[0])
        chosen = concrete[0]
        chosen_key = chosen[1]
        if chosen_key in DTI_CATEGORIES:
            return DTI_CATEGORIES[chosen_key], f"rule:{chosen[2]}", 1.0
        else:
            return str(chosen_key), f"rule:{chosen[2]}", 1.0

    # If only generic dealer/trading matched or no match, inspect tokens to decide
    tokens = set(re.findall(r"\b[\w\-]+\b", t))
    if tokens & ESSENTIAL_KEYWORDS:
        return DTI_CATEGORIES["Essential Retail"], "token_essential_keyword", 1.0

    # Dealer/trader: try to detect what they deal in
    if "dealer" in t or "trading" in t or "trader" in t or "wholesaler" in t or "wholesale" in t or "retailer" in t:
        hardware_terms = {"hardware", "lumber", "cement", "steel", "construction supply", "aggregates", "aggregate"}
        if any(h in t for h in hardware_terms):
            return DTI_CATEGORIES["Hardware Stores"], "dealer_item:hardware", 1.0
        if tokens & ESSENTIAL_KEYWORDS:
            return DTI_CATEGORIES["Essential Retail"], "dealer_item:essential", 1.0
        if any(x in t for x in ["motor parts", "auto supply", "auto parts", "car parts"]):
            return DTI_CATEGORIES["AutoRepair"], "dealer_item:auto", 1.0
        if any(x in t for x in ["veterinary", "animal", "feed", "fertilizer"]):
            return DTI_CATEGORIES["Veterinary Activities"], "dealer_item:vet", 1.0
        return DTI_CATEGORIES["Essential Retail"], "dealer_item:fallback_to_retail", 0.9

    # As a last resort use difflib semantic fallback
    match, score = best_difflib_match(t, cutoff=0.60)
    if match:
        return match, "difflib", float(round(score, 3))

    # Final default: Other Services
    return DTI_CATEGORIES["Other Services"], "default_other", 0.0

def detect_dti_violations(lob_text: str, dti_category: str):
    """
    Detect potential DTI subcategory violations based on business description.
    Returns: (violation_status, violation_details, violation_count)
    """
    text = (lob_text or "").strip().lower()
    dti_cat = dti_category.strip()
    
    if text == "" or dti_cat == "":
        return "No Data", "No business description or category", 0
    
    violations = []
    
    # Check if this DTI category has violation rules
    if dti_cat in VIOLATION_RULES:
        for pattern, violation_type in VIOLATION_RULES[dti_cat]:
            try:
                if re.search(pattern, text):
                    violations.append(violation_type)
            except re.error:
                continue
    
    # Check for general violations that apply to all categories
    general_violations = [
        (r"\b(illegal|unlicensed|unregistered|black market|underground)\b", "Unlicensed/Illegal operations"),
        (r"\b(minor|underage|below 18|below18)\b", "Minor employment/age restriction violation"),
        (r"\b(loan|lending|money lending|usury|5-6)\b", "Money lending operations without permit"),
        (r"\b(gambling|betting|casino|poker|jueteng|masiao)\b", "Gambling operations violation"),
    ]
    
    for pattern, violation_type in general_violations:
        try:
            if re.search(pattern, text):
                violations.append(violation_type)
        except re.error:
            continue
    
    if violations:
        return "Potential Violation", " | ".join(violations), len(violations)
    else:
        return "Compliant", "No violations detected", 0

# ---------------------------
# Crime Subviolation Mapping Functions
# ---------------------------

def normalize_crime_text(text: str) -> str:
    """Uppercase + collapse spaces for consistent matching."""
    t = str(text).upper().strip()
    t = re.sub(r"\s+", " ", t)
    return t

def load_crime_reference_data():
    """Load crime subviolation and violation reference data."""
    try:
        subviolation_df = pd.read_csv("static/crime_subviolation.csv")
        violation_df = pd.read_csv("static/crime_violation .csv")
        
        # Normalize keys for matching
        subviolation_df["SubviolationName"] = subviolation_df["name"].astype(str)
        subviolation_df["SubKey"] = subviolation_df["SubviolationName"].apply(normalize_crime_text)
        
        violation_df["ViolationName"] = violation_df["name"].astype(str)
        
        return subviolation_df, violation_df
    except Exception as e:
        print(f"Error loading crime reference data: {e}")
        return None, None

def map_crime_subviolation(main_df, target_column):
    """Map subviolations and violations to main crime data."""
    subviolation_df, violation_df = load_crime_reference_data()
    
    if subviolation_df is None or violation_df is None:
        return main_df, "Error loading reference data"
    
    # Debug: Print sample data
    print(f"Target column: {target_column}")
    print(f"Sample main data: {main_df[target_column].head(10).tolist()}")
    print(f"Sample subviolation names: {subviolation_df['name'].head(10).tolist()}")
    
    # Create key in main data
    main_df["SubKey"] = main_df[target_column].apply(normalize_crime_text)
    
    # Debug: Print normalized keys
    print(f"Sample main keys: {main_df['SubKey'].head(10).tolist()}")
    print(f"Sample subviolation keys: {subviolation_df['SubKey'].head(10).tolist()}")
    
    # Check for any matches
    main_keys = set(main_df["SubKey"].unique())
    sub_keys = set(subviolation_df["SubKey"].unique())
    matches = main_keys.intersection(sub_keys)
    print(f"Potential matches: {len(matches)}")
    if matches:
        print(f"Sample matches: {list(matches)[:5]}")
    
    # Merge with subviolation table
    merged = main_df.merge(
        subviolation_df[["id", "SubviolationName", "SubKey", "violation_id"]],
        on="SubKey",
        how="left"
    ).rename(columns={
        "SubviolationName": "Subviolation Name"
    })
    
    # Merge with violation table
    merged = merged.merge(
        violation_df[["id", "ViolationName"]],
        left_on="violation_id",
        right_on="id",
        how="left"
    ).rename(columns={
        "ViolationName": "Violation Name"
    })
    
    # Fill null genders if they exist
    gender_fields = ["Complainant Gender", "Respondent Gender"]
    for g in gender_fields:
        if g in merged.columns:
            merged[g] = merged[g].fillna("Unknown").replace("", "Unknown")
    
    # Add docket_number if not present
    if "docket_number" not in merged.columns:
        merged["docket_number"] = "NA"
    
    return merged, "Crime subviolation mapping completed successfully"

@app.route('/export_csv')
def export_csv():
    global uploaded_df
    if uploaded_df.empty:
        return "No data to export.", 400

    csv_buffer = StringIO()
    uploaded_df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    return Response(
        csv_buffer.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=cleaned_data.csv'}
    )

@app.route('/export_excel')
def export_excel():
    global uploaded_df
    if uploaded_df.empty:
        return "No data to export.", 400

    excel_buffer = BytesIO()
    uploaded_df.to_excel(excel_buffer, index=False, sheet_name='Cleaned Data')
    excel_buffer.seek(0)

    return send_file(
        excel_buffer,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name='cleaned_data.xlsx'
    )

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/load_csv', methods=['POST'])
def load_csv():
    global uploaded_df, history_stack

    if request.method == 'POST':
        file = request.files.get('file')
        if file is None:
            return "No file uploaded.", 400

        # Debug: Print file info
        print(f"File received: {file.filename}")
        print(f"File content type: {file.content_type}")
        
        # Check file extension
        filename = file.filename.lower()
        if not (filename.endswith('.csv') or filename.endswith('.xlsx')):
            print(f"Invalid file extension: {filename}")
            return "Invalid file format. Please upload a CSV or Excel (.xlsx) file.", 400

        try:
            if filename.endswith('.xlsx'):
                # Handle Excel file
                print("Reading Excel file...")
                uploaded_df = pd.read_excel(file, engine='openpyxl')
                print(f"Excel file loaded successfully. Shape: {uploaded_df.shape}")
            else:
                # Handle CSV file with encoding detection
                print("Reading CSV file...")
                raw_data = file.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding']
                file.seek(0)
                uploaded_df = pd.read_csv(file, encoding=encoding)
                print(f"CSV file loaded successfully. Shape: {uploaded_df.shape}")
            
            history_stack.clear()  # Clear history when loading new data
            return dataframe_to_html_with_id(uploaded_df)
        except Exception as e:
            print(f"Error reading file: {str(e)}")
            return f"Error reading file: {str(e)}", 400

    return render_template('index.html')

@app.route('/remove_special_chars')
def remove_special_chars():
    global uploaded_df
    save_history()
    
    def clean_special_chars(val):
        if isinstance(val, str):
            val_stripped = val.strip()
            
            # Check if it's a date/time pattern
            date_pattern = r'^\d{1,4}[/-]\d{1,2}[/-]\d{1,4}$'
            time_pattern = r'^\d{1,2}:\d{2}(:\d{2})?\s*(AM|PM|am|pm)?$'
            datetime_pattern = r'^\d{1,4}[/-]\d{1,2}[/-]\d{1,4}\s+\d{1,2}:\d{2}(:\d{2})?\s*(AM|PM|am|pm)?$'
            
            if re.match(date_pattern, val_stripped) or re.match(time_pattern, val_stripped) or re.match(datetime_pattern, val_stripped):
                return re.sub(r"[^0-9\s/:\-AMPampm]", '', val)
            
            # Check email patterns
            email_pattern = r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$'
            # More flexible pattern for malformed emails (including those without dots)
            loose_email_pattern = r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+$'
            
            if re.match(email_pattern, val_stripped):
                cleaned = re.sub(r"[^A-Za-z0-9@._%+\-]", '', val)
                return cleaned
            elif re.match(loose_email_pattern, val_stripped):
                # Loose email format - fix common issues
                cleaned = re.sub(r"@+", "@", val)  # Replace multiple @ with single @
                cleaned = re.sub(r"[^A-Za-z0-9@._%+\-]", '', cleaned)  # Remove other special chars
                return cleaned
            else:
                return re.sub(r"[^A-Za-z0-9\s\-',&@:\/]", '', val)
        return val

    uploaded_df = uploaded_df.applymap(clean_special_chars)
    return "<p><b>Removed special characters (preserved valid emails, dates, times, kept - ' , @ : /) from text columns.</b></p>" + dataframe_to_html_with_id(uploaded_df)

@app.route('/load_sample')
def load_sample():
    global uploaded_df, history_stack
    uploaded_df = pd.DataFrame({
        'Name': [
            '  john', 'Jane  ', 'JANE', 'John', 'Alice', 'alice ', ' Bob', 'bob', 'BOB', None,
            'Charlie', ' charlie ', 'Dana', 'dana', 'D@n@', 'Eve', 'eve', 'EV3', 'Mallory', ' mallory ',
            'Zoe', 'Zoe', 'ZOE ', '', 'Grace', ' grace', 'Oscar', 'Oscar', 'oscar', 'Oscar'
        ],
        'Age': [
            '25', 30, 'thirty', 25, None, 27, 31, '31', 31, 29,
            'NaN', 33, 45, 45, 38, 28, '28', None, 'thirty-two', 32,
            24, 24, '24 ', None, 'N/A', 36, 40, None, 40, 40
        ],
        'Gender': [
            'M', 'F', 'f', 'm', 'F', 'F', 'M', 'm', 'M', 'M',
            None, 'M', 'F', 'F', 'f', 'F', 'f', 'F', 'F', None,
            'Female', 'female', 'F ', 'M', '', 'F', 'Male', 'male', 'm', 'M'
        ],
        'Email': [
            'john@example.com', ' JANE@MAIL.COM ', 'jane@mail', 'john@', 'alice@email.com', 'alice@EMAIL.COM',
            'bob@sample.net', 'bob@@sample.com', 'bob@gmail.com', 'invalid-email',
            '', 'charlie@mail.com', 'dana@mail.com', 'dana@MAIL.COM', 'd@n@!mail.com',
            'eve@mail.com', 'EVE@MAIL.com', 'ev3@mail', 'mallory@mail.com', ' mallory@web.com ',
            'zoe@gmail', 'zoe@', 'zoe@example.com', None, 'grace@mail', ' grace@web.com', 'oscar@site', '', 'oscar@site.com', 'OSCAR@site.com'
        ],
        'JoinDate': [
            '2020-01-15', '01/15/2020', '15-Jan-2020', '2020/01/15', 'Jan 15 2020', '2020.01.15',
            '20200115', '15/01/2020', '2020-13-01', 'Not a date',
            '', '2020-01-15', '2021-02-30', '2022/02/28', 'Feb 28, 2022',
            '03-05-2021', '2021/05/03', 'May 3rd, 2021', None, '2021-05-03',
            '2021-05-03', '2021-5-3', '03 May 2021', 'N/A', 'null', '2022-06-15', '2022-06-15', '', '2022-06-15', '2022-06-15'
        ],
        'Salary': [
            '50000', '60,000', '50K', '55000.50', '$58000', None,
            '59000', 'invalid', '', '52000',
            '48000', 'NaN', '61000', '62000', '63,000',
            '64000', '65000', '66000', '67000', 'not available',
            '68000', '68000.00', '69000', '70000', '70000', '71000', '72000', '0', None, '73000'
        ],
        'Active': [
            'Yes', 'No', 'yes', 'no', 'TRUE', 'FALSE',
            'true', 'false', 1, 0,
            'Y', 'N', '', '1', '0',
            'Active', 'Inactive', None, 'yes', 'no',
            'enabled', 'disabled', 'Yes', 'No', 'n/a', 'True', 'False', '', '1', '0'
        ]
    })
    history_stack.clear()
    return dataframe_to_html_with_id(uploaded_df)


@app.route('/remove_duplicates_by_column')
def remove_duplicates_by_column():
    global uploaded_df
    column = request.args.get('column')

    if not column:
        return "Error: Column parameter is required.", 400
    if column not in uploaded_df.columns:
        return f"Error: Column '{column}' does not exist in the data.", 400

    save_history()
    before = len(uploaded_df)
    
    # Create a normalized version of the column for duplicate detection
    # Convert to string, strip whitespace, and convert to lowercase
    normalized_series = uploaded_df[column].astype(str).str.strip().str.lower()
    
    # Find duplicates based on normalized values (keep first occurrence)
    duplicate_mask = normalized_series.duplicated(keep='first')
    
    # Remove duplicates
    uploaded_df = uploaded_df[~duplicate_mask]
    
    # Reset index to ensure continuous row IDs
    uploaded_df = uploaded_df.reset_index(drop=True)
    
    after = len(uploaded_df)
    affected = before - after
    return f"<p><b>Removed {affected} duplicate rows based on column '{column}' (case-insensitive, whitespace normalized).</b></p>" + dataframe_to_html_with_id(uploaded_df)

@app.route('/standardize_text')
def standardize_text():
    global uploaded_df
    save_history()
    uploaded_df = uploaded_df.applymap(lambda x: x.title() if isinstance(x, str) else x)
    return dataframe_to_html_with_id(uploaded_df)

@app.route('/missing_values')
def missing_values():
    global uploaded_df
    missing_rows = uploaded_df[uploaded_df.isnull().any(axis=1)]
    return missing_rows.to_json(orient='records')

@app.route('/drop_missing')
def drop_missing():
    global uploaded_df
    save_history()
    before = len(uploaded_df)
    uploaded_df = uploaded_df.dropna()
    after = len(uploaded_df)
    affected = before - after
    return f"<p><b>Dropped {affected} rows with missing values.</b></p>" + dataframe_to_html_with_id(uploaded_df)

@app.route('/fill_missing')
def fill_missing():
    global uploaded_df
    save_history()
    
    print("=== DEBUG: Fill Missing Function ===")
    print(f"DataFrame shape: {uploaded_df.shape}")
    print(f"Data types:\n{uploaded_df.dtypes}")
    print(f"Sample values before:\n{uploaded_df.head(10)}")
    
    before_missing = uploaded_df.isnull().sum().sum()
    print(f"True null/NaN count: {before_missing}")
    
    # Count string-based missing values before replacement
    string_missing_count = 0
    missing_indicators = ['', 'NaN', 'nan', 'null', 'NULL', 'N/A', 'n/a', 'None', 'none', '-', '--', 'missing', 'Missing']
    
    for col in uploaded_df.columns:
        for indicator in missing_indicators:
            # Count exact matches
            matches = (uploaded_df[col].astype(str) == indicator).sum()
            if matches > 0:
                print(f"Column '{col}': Found {matches} matches for '{indicator}'")
                string_missing_count += matches
                uploaded_df[col] = uploaded_df[col].replace(indicator, 0)
    
    print(f"String-based missing count: {string_missing_count}")
    
    # Handle actual None/NaN values
    uploaded_df = uploaded_df.fillna(0)
    
    after_missing = uploaded_df.isnull().sum().sum()
    filled_nulls = before_missing - after_missing
    total_filled = filled_nulls + string_missing_count
    
    print(f"Values after fillna:\n{uploaded_df.head(10)}")
    print(f"Total filled: {total_filled}")
    print("=== END DEBUG ===")
    
    return f"<p><b>Filled {total_filled} missing cells with 0 (including {string_missing_count} string-based missing values).</b></p>" + dataframe_to_html_with_id(uploaded_df)

@app.route('/fix_dates')
def fix_dates():
    global uploaded_df
    save_history()
    
    def standardize_date(val):
        if pd.isnull(val) or val == '':
            return val
        
        try:
            # Convert to string and strip
            val_str = str(val).strip()
            
            # Skip obvious non-dates
            if val_str.lower() in ['not a date', 'n/a', 'null', 'nan', 'none']:
                return val
            
            # Try parsing with dateutil.parser (very flexible)
            parsed_date = parser.parse(val_str, fuzzy=True, dayfirst=False)
            
            # Return in standard format
            return parsed_date.strftime('%Y-%m-%d')
            
        except:
            # If parsing fails, return original value
            return val
    
    # Apply to all columns that might contain dates
    date_columns = []
    for col in uploaded_df.columns:
        col_lower = col.lower()
        # Check if column might be a date column
        if any(keyword in col_lower for keyword in ['date', 'time', 'created', 'updated', 'join', 'start', 'end']):
            date_columns.append(col)
    
    fixed_count = 0
    for col in date_columns:
        original_values = uploaded_df[col].copy()
        uploaded_df[col] = uploaded_df[col].apply(standardize_date)
        # Count how many were actually changed
        changes = (original_values != uploaded_df[col]).sum()
        fixed_count += changes
    
    if date_columns:
        return f"<p><b>Fixed {fixed_count} date values in columns: {', '.join(date_columns)} to YYYY-MM-DD format.</b></p>" + dataframe_to_html_with_id(uploaded_df)
    else:
        return f"<p><b>No date columns detected. Looked for columns containing: date, time, created, updated, join, start, end.</b></p>" + dataframe_to_html_with_id(uploaded_df)

@app.route('/sort_values')
def sort_values():
    global uploaded_df
    column = request.args.get('column')

    if column is None or column not in uploaded_df.columns.astype(str).tolist():
        return f"Invalid column name: {html.escape(str(column))}", 400

    save_history()
    uploaded_df.sort_values(by=column, inplace=True)
    return f"<p><b>Sorted by '{html.escape(str(column))}'.</b></p>" + dataframe_to_html_with_id(uploaded_df)

@app.route('/sort_interface')
def sort_interface():
    global uploaded_df
    if uploaded_df.empty:
        return "No data loaded.", 400

    uploaded_df.columns = uploaded_df.columns.map(lambda x: str(x) if x is not None else "Unnamed")
    options_html = ''.join(
        f'<option value="{html.escape(col)}">{html.escape(col)}</option>'
        for col in uploaded_df.columns
    )

    return f'''
        <form action="/sort_values" method="get">
            <label for="column">Sort by column:</label>
            <select name="column" id="column">
                {options_html}
            </select>
            <button type="submit">Sort</button>
        </form>
        <br>
        {dataframe_to_html_with_id(uploaded_df)}
    '''

@app.route('/bar_chart')
def bar_chart():
    global uploaded_df
    plt.figure()
    uploaded_df[uploaded_df.columns[0]].value_counts().plot(kind='bar')
    plt.title('Bar Chart')
    path = os.path.join('static', 'bar_chart.png')
    plt.savefig(path)
    plt.close()
    return "Bar chart generated."

@app.route('/pie_chart')
def pie_chart():
    global uploaded_df
    plt.figure()
    uploaded_df[uploaded_df.columns[0]].value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title('Pie Chart')
    path = os.path.join('static', 'pie_chart.png')
    plt.savefig(path)
    plt.close()
    return "Pie chart generated."

@app.route('/change_dtype')
def change_dtype():
    global uploaded_df
    column = request.args.get('column')
    dtype = request.args.get('dtype')

    if column not in uploaded_df.columns:
        return f"Invalid column: {column}", 400

    save_history()
    try:
        if dtype in ['datetime', 'date', 'time']:
            # Handle datetime conversions
            if dtype == 'datetime':
                uploaded_df[column] = pd.to_datetime(uploaded_df[column], errors='coerce')
            elif dtype == 'date':
                uploaded_df[column] = pd.to_datetime(uploaded_df[column], errors='coerce').dt.date
            elif dtype == 'time':
                # For time, try to extract time from datetime or parse time strings
                temp_series = pd.to_datetime(uploaded_df[column], errors='coerce')
                # If conversion failed, try parsing as time directly
                if temp_series.isna().all():
                    uploaded_df[column] = pd.to_datetime(uploaded_df[column], format='mixed', errors='coerce').dt.time
                else:
                    uploaded_df[column] = temp_series.dt.time
        elif dtype in ['int', 'float']:
            # Handle numeric conversions with better error handling
            if dtype == 'int':
                # First convert to float to handle decimals, then to int
                temp_series = pd.to_numeric(uploaded_df[column], errors='coerce')
                uploaded_df[column] = temp_series.astype('Int64')  # Nullable integer
            elif dtype == 'float':
                uploaded_df[column] = pd.to_numeric(uploaded_df[column], errors='coerce')
        elif dtype in ['bool']:
            # Handle boolean conversion
            def to_bool(val):
                if pd.isna(val):
                    return False
                if isinstance(val, str):
                    val_lower = val.lower().strip()
                    return val_lower in ['true', '1', 'yes', 'y', 'active', 'enabled']
                return bool(val)
            uploaded_df[column] = uploaded_df[column].apply(to_bool)
        else:
            # Handle standard pandas dtypes
            uploaded_df[column] = uploaded_df[column].astype(dtype)
            
    except Exception as e:
        return f"Conversion failed: {e}", 400

    # Count how many values were converted vs became null
    total_count = len(uploaded_df[column])
    null_count = uploaded_df[column].isna().sum()
    success_count = total_count - null_count
    
    return f"<p><b>Changed '{column}' to {dtype}. Successfully converted {success_count}/{total_count} values ({null_count} became null).</b></p>" + dataframe_to_html_with_id(uploaded_df)

@app.route('/replace_values')
def replace_values_non_sensitive():
    """Non-sensitive replace: substring replacement (case-insensitive)."""
    global uploaded_df
    column = request.args.get('column')
    to_replace = request.args.get('replace_from', '')
    new_value = request.args.get('to', '')

    if column not in uploaded_df.columns:
        return f"Invalid column: {column}", 400

    save_history()

    def replace_substring(val):
        if isinstance(val, str):
            return re.sub(re.escape(to_replace), new_value, val, flags=re.IGNORECASE)
        # Handle numeric values - convert to string for comparison
        val_str = str(val)
        if re.search(re.escape(to_replace), val_str, flags=re.IGNORECASE):
            return new_value
        return val

    # Count affected rows
    before_count = uploaded_df[column].apply(
        lambda x: isinstance(x, str) and re.search(re.escape(to_replace), x, flags=re.IGNORECASE) is not None
    ).sum()
    # Add numeric matches
    before_count += uploaded_df[column].apply(
        lambda x: not isinstance(x, str) and re.search(re.escape(to_replace), str(x), flags=re.IGNORECASE) is not None
    ).sum()

    uploaded_df[column] = uploaded_df[column].apply(replace_substring)

    return f"<p><b>Non-sensitive replace (substring): '{to_replace}' → '{new_value}' in '{column}'. Rows affected: {before_count}</b></p>" + dataframe_to_html_with_id(uploaded_df)


@app.route('/replace_values_sensitive')
def replace_values_sensitive():
    """Sensitive replace: full-value replacement only (normalized)."""
    global uploaded_df
    import re

    column = request.args.get('column')
    to_replace = request.args.get('replace_from', '')
    new_value = request.args.get('to', '')

    if column not in uploaded_df.columns:
        return f"Invalid column: {column}", 400

    save_history()

    def normalize(text):
        if not isinstance(text, str):
            # Convert numeric values to string for normalization
            text = str(text)
        return re.sub(r'\s+', ' ', text).strip().lower()

    # Count affected rows
    before_count = (uploaded_df[column].apply(normalize) == normalize(to_replace)).sum()

    # Replace only if full value matches
    uploaded_df[column] = uploaded_df[column].apply(
        lambda x: new_value if normalize(x) == normalize(to_replace) else x
    )

    return f"<p><b>Sensitive replace (full match): '{to_replace}' → '{new_value}' in '{column}'. Rows affected: {before_count}</b></p>" + dataframe_to_html_with_id(uploaded_df)


@app.route('/rename_column')
def rename_column():
    global uploaded_df
    old_name = request.args.get('old')
    new_name = request.args.get('new')

    if old_name not in uploaded_df.columns:
        return f"Column '{old_name}' not found.", 400

    save_history()
    uploaded_df.rename(columns={old_name: new_name}, inplace=True)
    return f"<p><b>Renamed '{old_name}' to '{new_name}'.</b></p>" + dataframe_to_html_with_id(uploaded_df)

@app.route('/undo_replace')
def undo_replace():
    global uploaded_df, history_stack
    if history_stack:
        uploaded_df = history_stack.pop()
        return "<p><b>Undo successful.</b></p>" + dataframe_to_html_with_id(uploaded_df)
    else:
        return "Nothing to undo.", 400

@app.route('/drop_column')
def drop_column():
    global uploaded_df
    column = request.args.get('column')

    if column not in uploaded_df.columns:
        return f"Column '{column}' not found.", 400

    save_history()
    uploaded_df.drop(columns=[column], inplace=True)
    return f"<p><b>Dropped column '{column}'.</b></p>" + dataframe_to_html_with_id(uploaded_df)

@app.route('/drop_value')
def drop_value():
    global uploaded_df
    column = request.args.get('column')
    value = request.args.get('value', "").strip()
    action = request.args.get('action', 'drop')  # default = drop

    if column not in uploaded_df.columns:
        return f"Column '{column}' not found.", 400

    save_history()

    # Ensure it's a string column for matching
    uploaded_df[column] = uploaded_df[column].astype(str)

    # Build mask: case-insensitive substring search
    mask = uploaded_df[column].str.contains(value, case=False, na=False)

    if action == "drop":
        uploaded_df = uploaded_df[~mask]  # drop rows that match
        message = f"Dropped rows where {column} contains '{value}' (case-insensitive)"
    elif action == "keep":
        uploaded_df = uploaded_df[mask]   # keep only matching rows
        message = f"Kept rows where {column} contains '{value}' (case-insensitive)"
    else:
        return "Invalid action", 400

    return f"<p><b>{message}</b></p>" + dataframe_to_html_with_id(uploaded_df)


@app.route('/data_profile')
def data_profile():
    global uploaded_df
    if uploaded_df.empty:
        return "No data loaded.", 400

    profile = {}
    profile['Total Records'] = len(uploaded_df)

    for col in uploaded_df.columns:
        col_lower = col.lower()
        series = uploaded_df[col]

        # Missing values
        missing_count = series.isnull().sum()
        if missing_count > 0:
            profile[f"Missing values in '{col}'"] = missing_count

        # Invalid phone numbers
        if 'phone' in col_lower:
            pattern = r'^\+?\d{10,15}$'
            invalid_phones = series.apply(lambda x: not bool(re.fullmatch(pattern, str(x))) if pd.notnull(x) else False).sum()
            profile[f"Invalid Phone Numbers in '{col}'"] = invalid_phones

        # Invalid emails
        if 'email' in col_lower:
            pattern = r'^[^@\s]+@[^@\s]+\.[^@\s]+$'
            invalid_emails = series.apply(lambda x: not bool(re.fullmatch(pattern, str(x))) if pd.notnull(x) else False).sum()
            profile[f"Invalid Emails in '{col}'"] = invalid_emails

        # Non-standard country codes
        if 'country' in col_lower:
            nonstandard = series.apply(lambda x: not bool(re.fullmatch(r'^[A-Za-z]{2,3}$', str(x))) if pd.notnull(x) else False).sum()
            profile[f"Non-standard Country Codes in '{col}'"] = nonstandard

        # Missing IDs
        if 'id' in col_lower:
            missing_ids = series.isnull().sum()
            profile[f"Missing IDs in '{col}'"] = missing_ids

        # Duplicates per column (excluding nulls)
        duplicate_vals = series[series.notnull()].duplicated().sum()
        if duplicate_vals > 0:
            profile[f"Duplicate Values in '{col}'"] = duplicate_vals

    profile['Duplicate Rows (Full Match)'] = uploaded_df.duplicated().sum()

    # Styled HTML with close button
    html_output = """
    <html>
    <head>
        <style>
            body { font-family: Arial, sans-serif; background: #f9f9f9; padding: 20px; }
            table { border-collapse: collapse; width: 100%; margin-top: 10px; background: white; }
            th, td { border: 1px solid #ccc; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            #profile-container { border: 1px solid #ccc; padding: 15px; background: #fff; border-radius: 8px; position: relative; }
            .close-btn {
                position: absolute;
                top: 10px;
                right: 10px;
                background: red;
                color: white;
                border: none;
                padding: 5px 10px;
                cursor: pointer;
                border-radius: 4px;
            }
        </style>
    </head>
    <body><br>
        <div id="profile-container">
            <h3>Input Data Profile (Pre-Cleaning)</h3>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
    """

    for key, value in profile.items():
        html_output += f"<tr><td>{key}</td><td>{value}</td></tr>"

    html_output += """
            </table>
        </div>

        <!-- Place the script AFTER the HTML it interacts with -->
        <script>
            function closeProfile() {
                const container = document.getElementById('profile-container');
                if (container) container.style.display = 'none';
            }
        </script>
    </body>
    </html>
    """

    return html_output


@app.route('/generate_alias', methods=['POST'])
def generate_alias():
    global uploaded_df
    if uploaded_df.empty:
        return "No data loaded.", 400

    prefix = request.form.get("prefix", "").strip()
    start_num = request.form.get("start", "").strip()
    target_col = request.form.get("column", "").strip()

    if not prefix or not start_num or not target_col:
        return "Missing inputs.", 400

    if target_col not in uploaded_df.columns:
        # auto-create column
        uploaded_df[target_col] = ""

    # Get total row
    total = len(uploaded_df)

    # Compute digit length automatically
    digit_len = len(start_num)

    try:
        start_int = int(start_num)
    except:
        return "Invalid starting number format.", 400

    save_history()

    # Generate aliases
    aliases = []
    for i in range(total):
        number_str = str(start_int + i).zfill(digit_len)
        aliases.append(f"{prefix}{number_str}")

    uploaded_df[target_col] = aliases

    return "<p><b>Alias generated successfully!</b></p>" + dataframe_to_html_with_id(uploaded_df)


@app.route('/find')
def find():
    global uploaded_df
    keyword = request.args.get('keyword')
    column = request.args.get('column')

    if not keyword:
        return "Please provide a keyword using ?keyword=your_value", 400

    if uploaded_df.empty:
        return "No data loaded.", 400

    if column:
        if column not in uploaded_df.columns:
            return f"Column '{column}' not found.", 400
        mask = uploaded_df[column].astype(str).str.contains(keyword, case=False, na=False)
        filtered_df = uploaded_df[mask]
    else:
        mask = uploaded_df.apply(lambda row: row.astype(str).str.contains(keyword, case=False, na=False).any(), axis=1)
        filtered_df = uploaded_df[mask]

    if filtered_df.empty:
        return f"No matching records found for '{keyword}'."

    return f"<p><b>Found {len(filtered_df)} matching rows for keyword '{keyword}'</b></p>" + filtered_df.to_html()

from word2number import w2n

@app.route('/magic_clean')
def magic_clean():
    global uploaded_df
    save_history()

    def clean_column_names(df):
        df.columns = df.columns.str.strip().str.lower().str.replace(r'\s+', '_', regex=True)
        return df

    def clean_value(val, col_name):
        if pd.isnull(val):
            return None

        if isinstance(val, str):
            original_val = val.strip()
            val = original_val  # keep original as backup

            # Clean email fields
            if 'email' in col_name:
                val = re.sub(r'@+', '@', val)
                if '@' in val and not re.search(r'\.\w+$', val):
                    val += '.com'
                if re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', val):
                    return val.lower()
                else:
                    return None

            # Clean boolean-like values
            if val.lower() in ['yes', 'true', 'y', '1', 'active', 'enabled']:
                return 1
            elif val.lower() in ['no', 'false', 'n', '0', 'inactive', 'disabled']:
                return 0

            # Try to convert word numbers
            try:
                word_to_num = w2n.word_to_num(val.lower())
                return word_to_num
            except:
                pass

            # Clean numeric values (with k/m symbols, $, commas)
            val_numeric = val.lower().replace(',', '').replace('$', '')
            if val_numeric.endswith('k'):
                try:
                    return int(float(val_numeric.replace('k', '')) * 1000)
                except:
                    pass
            elif val_numeric.endswith('m'):
                try:
                    return int(float(val_numeric.replace('m', '')) * 1_000_000)
                except:
                    pass
            elif re.fullmatch(r'-?\d+(\.\d+)?', val_numeric):
                return float(val_numeric) if '.' in val_numeric else int(val_numeric)

            # Try parsing as a date
            try:
                parsed = parser.parse(val, fuzzy=True, dayfirst=False)
                return parsed.strftime('%Y-%m-%d')
            except:
                pass

            # If column name likely refers to a name or title, standardize it
            if any(key in col_name for key in ['name', 'title', 'position']):
                val = re.sub(r'[^A-Za-zÀ-ÿ\'\- ]+', '', val)
                return val.title().strip()

            # For other text columns, keep original (may contain valid punctuation/numbers)
            return original_val

        # Convert Python boolean to 1/0
        if isinstance(val, bool):
            return int(val)

        return val

    uploaded_df = clean_column_names(uploaded_df)
    uploaded_df = uploaded_df.apply(lambda col: col.apply(lambda val: clean_value(val, col.name)))
    uploaded_df.drop_duplicates(inplace=True)

    return "<p><b>✨ Magic Clean applied!</b></p>" + dataframe_to_html_with_id(uploaded_df)

@app.route('/generate_chart_data')
def generate_chart_data():
    global uploaded_df
    x_col = request.args.get('x')
    y_col = request.args.get('y')

    if uploaded_df.empty or x_col not in uploaded_df.columns or y_col not in uploaded_df.columns:
        return jsonify({'labels': [], 'values': []})

    # Drop missing values
    df = uploaded_df[[x_col, y_col]].dropna()
    
    # Check if Y column is numeric
    try:
        y_numeric = pd.to_numeric(df[y_col], errors='coerce')
        if y_numeric.notna().any():
            # Y column has numeric data, use aggregation
            df[y_col] = y_numeric
            df = df.dropna()
            grouped = df.groupby(x_col)[y_col].sum().reset_index()
            labels = grouped[x_col].astype(str).tolist()
            values = grouped[y_col].tolist()
        else:
            # Y column is categorical, use value counts
            grouped = df.groupby(x_col).size().reset_index(name='count')
            labels = grouped[x_col].astype(str).tolist()
            values = grouped['count'].tolist()
    except Exception as e:
        # Fallback to simple value counts
        try:
            grouped = df.groupby(x_col).size().reset_index(name='count')
            labels = grouped[x_col].astype(str).tolist()
            values = grouped['count'].tolist()
        except:
            return jsonify({'labels': [], 'values': []})

    return jsonify({'labels': labels, 'values': values})

@app.route('/upload_mapping_csv', methods=['POST'])
def upload_mapping_csv():
    global mapping_df
    file = request.files.get('file')
    if file is None:
        return "No mapping file uploaded.", 400

    # Check file extension
    filename = file.filename.lower()
    if not (filename.endswith('.csv') or filename.endswith('.xlsx')):
        return "Invalid file format. Please upload a CSV or Excel (.xlsx) file.", 400

    try:
        if filename.endswith('.xlsx'):
            # Handle Excel file
            mapping_df = pd.read_excel(file, engine='openpyxl')
        else:
            # Handle CSV file with encoding detection
            raw_data = file.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']
            file.seek(0)
            mapping_df = pd.read_csv(file, encoding=encoding)
        
        return mapping_df.to_html()
    except Exception as e:
        return f"Error reading mapping file: {str(e)}", 400

@app.route('/apply_dti_mapping')
def apply_dti_mapping():
    global uploaded_df

    if uploaded_df.empty:
        return "No data loaded.", 400

    # Get target column from query parameter
    target_col = request.args.get("target_col")
    if not target_col:
        return "Missing 'target_col' query parameter. Please select a column.", 400

    if target_col not in uploaded_df.columns:
        return f"Column '{target_col}' does not exist in the data.", 400

    save_history()
    
    # Create LOB_clean column
    uploaded_df["LOB_clean"] = uploaded_df[target_col].apply(normalize_text)
    
    # Apply DTI mapping
    mapped = uploaded_df["LOB_clean"].apply(lambda x: pd.Series(map_lob_to_dti(x), index=["DTI_Sub_Category", "Match_Method", "Match_Score"]))
    uploaded_df = pd.concat([uploaded_df, mapped], axis=1)
    
    # Apply violation detection
    violation_results = uploaded_df.apply(lambda row: pd.Series(detect_dti_violations(row["LOB_clean"], row["DTI_Sub_Category"]), 
                                                          index=["Violation_Status", "Violation_Details", "Violation_Count"]), axis=1)
    uploaded_df = pd.concat([uploaded_df, violation_results], axis=1)
    
    # Generate summary
    violation_summary = uploaded_df["Violation_Status"].value_counts().to_dict()
    total_violations = len(uploaded_df[uploaded_df['Violation_Status'] == 'Potential Violation'])
    total_compliant = len(uploaded_df[uploaded_df['Violation_Status'] == 'Compliant'])
    
    summary_html = f"""
    <div class="alert alert-info">
        <h6><i class="bi bi-info-circle"></i> DTI Mapping & Violation Detection Summary</h6>
        <p><strong>Total Businesses:</strong> {len(uploaded_df)}</p>
        <p><strong>Potential Violations:</strong> <span class="text-danger">{total_violations}</span></p>
        <p><strong>Compliant:</strong> <span class="text-success">{total_compliant}</span></p>
        <p><strong>Target Column:</strong> {target_col}</p>
    </div>
    """
    
    return summary_html + dataframe_to_html_with_id(uploaded_df)

@app.route('/apply_dti_subcategory_only')
def apply_dti_subcategory_only():
    global uploaded_df

    if uploaded_df.empty:
        return "No data loaded.", 400

    # Get target column from query parameter
    target_col = request.args.get("target_col")
    if not target_col:
        return "Missing 'target_col' query parameter. Please select a column.", 400

    if target_col not in uploaded_df.columns:
        return f"Column '{target_col}' does not exist in the data.", 400

    save_history()
    
    # Check if DTI Sub Category already exists
    if "DTI Sub Category" in uploaded_df.columns:
        return dataframe_to_html_with_id(uploaded_df)
    
    # Apply DTI mapping and extract only the DTI Sub Category
    uploaded_df["DTI Sub Category"] = uploaded_df[target_col].apply(lambda x: map_lob_to_dti(str(x))[0])
    
    # Generate summary
    summary_html = f"""
    <div class="alert alert-success">
        <h6><i class="bi bi-check-circle"></i> DTI Sub Category Mapping Complete</h6>
        <p><strong>Total Businesses:</strong> {len(uploaded_df)}</p>
        <p><strong>Target Column:</strong> {target_col}</p>
        <p><strong>Column Added:</strong> DTI Sub Category</p>
    </div>
    """
    
    return summary_html + dataframe_to_html_with_id(uploaded_df)

@app.route('/apply_crime_mapping')
def apply_crime_mapping():
    global uploaded_df

    if uploaded_df.empty:
        return "No data loaded.", 400

    # Get target column from query parameter
    target_col = request.args.get("target_col")
    if not target_col:
        return "Missing 'target_col' query parameter. Please select a column.", 400

    if target_col not in uploaded_df.columns:
        return f"Column '{target_col}' does not exist in the data.", 400

    save_history()
    
    # Apply crime subviolation mapping
    mapped_df, message = map_crime_subviolation(uploaded_df.copy(), target_col)
    
    if "Error" in message:
        return message, 400
    
    uploaded_df = mapped_df
    
    # Generate summary
    mapped_count = len(uploaded_df[uploaded_df['Subviolation Name'].notna()])
    total_count = len(uploaded_df)
    
    summary_html = f"""
    <div class="alert alert-success">
        <h6><i class="bi bi-check-circle"></i> Crime Subviolation Mapping Complete</h6>
        <p><strong>Total Records:</strong> {total_count}</p>
        <p><strong>Successfully Mapped:</strong> {mapped_count}</p>
        <p><strong>Target Column:</strong> {target_col}</p>
    </div>
    """
    
    return summary_html + dataframe_to_html_with_id(uploaded_df)

@app.route('/apply_subviolation_mapping')
def apply_subviolation_mapping():
    global uploaded_df, mapping_df

    if uploaded_df.empty or mapping_df.empty:
        return "Main data or mapping data is not loaded.", 400

    target_col = request.args.get("target_col")
    if not target_col:
        return "Missing 'target_col' query parameter.", 400

    if target_col not in uploaded_df.columns:
        return f"Column '{target_col}' does not exist in uploaded data.", 400

    if 'id' not in mapping_df.columns or 'name' not in mapping_df.columns:
        return "Mapping file must have 'id' and 'name' columns.", 400

    save_history()

    uploaded_df[target_col] = uploaded_df[target_col].map(
        dict(zip(mapping_df['id'], mapping_df['name']))
    )

    return f"<p><b>Applied mapping on column <code>{target_col}</code>.</b></p>" + dataframe_to_html_with_id(uploaded_df)

@app.route('/upload_barangay_csv', methods=['POST'])
def upload_barangay_csv():
    global barangay_df

    file = request.files.get('file')
    if file is None:
        return "No barangay file uploaded.", 400

    # Check file extension
    filename = file.filename.lower()
    if not (filename.endswith('.csv') or filename.endswith('.xlsx')):
        return "Invalid file format. Please upload a CSV or Excel (.xlsx) file.", 400

    try:
        if filename.endswith('.xlsx'):
            # Handle Excel file
            barangay_df = pd.read_excel(file, engine='openpyxl', header=None, names=["Barangay"])
        else:
            # Handle CSV file with encoding detection
            raw_data = file.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']
            file.seek(0)

            # Just load barangays exactly as they are
            barangay_df = pd.read_csv(file, encoding=encoding, header=None, names=["Barangay"])

        return dataframe_to_html_with_id(barangay_df)
    except Exception as e:
        return f"Error reading barangay file: {str(e)}", 400

@app.route('/apply_barangay_mapping', methods=['POST'])
def apply_barangay_template():
    global uploaded_df, barangay_df

    if uploaded_df.empty:
        return "No main CSV loaded.", 400
    if barangay_df.empty:
        return "No barangay template loaded.", 400

    # Get target column from request
    target_column = request.form.get('targetColumn')
    if not target_column:   
        return "Target column not provided.", 400
    if target_column not in uploaded_df.columns:
        return f"'{target_column}' column not found in main CSV.", 400

    # Prepare barangay template: remove parentheses for matching
    barangay_list = []
    barangay_map = {}  # maps simplified name → full template
    for b in barangay_df['Barangay']:
        if isinstance(b, str):
            b_clean = re.sub(r'\s*\(.*?\)\s*', '', b).strip().upper()  # remove anything in parentheses
            barangay_list.append(b_clean)
            barangay_map[b_clean] = b  # store original template

    def map_to_barangay(value):
        if not isinstance(value, str) or not value.strip():
            return value  # keep as-is if empty or invalid

        value_upper = value.upper()
        for barangay_clean in barangay_list:
            if barangay_clean in value_upper:
                return barangay_map[barangay_clean]  # return full template with (Pob.) if exists
        return value  # keep original if no match

    save_history()
    uploaded_df[target_column] = uploaded_df[target_column].apply(map_to_barangay)

    return f"<p><b>Mapped '{target_column}' to Barangay (template-aware) for {len(uploaded_df)} rows.</b></p>" + dataframe_to_html_with_id(uploaded_df)

@app.route('/show_unaligned_barangays')
def show_unaligned_barangays():
    global uploaded_df, barangay_df

    if uploaded_df.empty:
        return "No main CSV loaded.", 400
    if barangay_df.empty:
        return "No barangay template loaded.", 400
    if 'ConsumerAddress' not in uploaded_df.columns:
        return "'ConsumerAddress' column not found in main CSV.", 400

    # Ensure all addresses are strings
    uploaded_df['ConsumerAddress'] = uploaded_df['ConsumerAddress'].astype(str).str.strip()

    # Get template barangay list (both original and cleaned versions)
    template_list = barangay_df['Barangay'].astype(str).str.strip().tolist()
    
    # Also create cleaned version for matching
    cleaned_template = []
    for b in barangay_df['Barangay']:
        if isinstance(b, str):
            b_clean = re.sub(r'\s*\(.*?\)\s*', '', b).strip().upper()
            cleaned_template.append(b_clean)

    # Find unaligned addresses (not in template or cleaned template)
    def is_aligned(address):
        if not isinstance(address, str) or not address.strip():
            return False
        
        address_upper = address.upper().strip()
        # Check if exact match in template
        if address in template_list:
            return True
        # Check if matches any cleaned template
        for template_clean in cleaned_template:
            if template_clean in address_upper:
                return True
        return False
    
    unaligned = uploaded_df[~uploaded_df['ConsumerAddress'].apply(is_aligned)]

    if unaligned.empty:
        return "<p><b>All addresses are aligned with the template.</b></p>"

    return (
        f"<p><b>Found {len(unaligned)} unaligned addresses:</b></p>" 
        + unaligned.to_html()
    )

@app.route('/export_unaligned_barangays')
def export_and_drop_unaligned_barangays():
    global uploaded_df, barangay_df

    # Get target column dynamically from query string
    target_column = request.args.get('targetColumn', 'ConsumerAddress').strip()

    if uploaded_df.empty:
        return "No main CSV loaded.", 400
    if barangay_df.empty:
        return "No barangay template loaded.", 400
    if target_column not in uploaded_df.columns:
        return f"'{target_column}' column not found in main CSV.", 400

    # Ensure all addresses are strings
    uploaded_df[target_column] = uploaded_df[target_column].astype(str).str.strip()
    
    # Get template barangay list (both original and cleaned versions)
    template_list = barangay_df['Barangay'].astype(str).str.strip().tolist()
    
    # Also create cleaned version for matching
    cleaned_template = []
    for b in barangay_df['Barangay']:
        if isinstance(b, str):
            b_clean = re.sub(r'\s*\(.*?\)\s*', '', b).strip().upper()
            cleaned_template.append(b_clean)

    # Find unaligned addresses (not in template or cleaned template)
    def is_aligned(address):
        if not isinstance(address, str) or not address.strip():
            return False
        
        address_upper = address.upper().strip()
        # Check if exact match in template
        if address in template_list:
            return True
        # Check if matches any cleaned template
        for template_clean in cleaned_template:
            if template_clean in address_upper:
                return True
        return False
    
    unaligned = uploaded_df[~uploaded_df[target_column].apply(is_aligned)]

    if unaligned.empty:
        return "<p><b>No unaligned addresses to export or drop.</b></p>"

    # Export CSV in-memory
    output = io.StringIO()
    unaligned.to_csv(output, index=False)
    output.seek(0)

    # RETAIN MAPPED BARANGAYS - Drop only the unaligned rows
    aligned = uploaded_df[uploaded_df[target_column].apply(is_aligned)]
    uploaded_df = aligned  # Keep only the aligned/mapped barangays

    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name='unaligned_barangays.csv'
    )

    

@app.route('/bulk_delete_rows', methods=['POST'])
def bulk_delete_rows():
    global uploaded_df
    try:
        data = request.get_json()
        row_ids = data.get('row_ids', [])
        
        if not row_ids:
            return "No row IDs provided.", 400
            
        # Convert string IDs to integers (they come from data-id attributes)
        row_indices = [int(id) for id in row_ids]
        
        # Save history before deletion
        save_history()
        
        # Get original count
        before_count = len(uploaded_df)
        
        # Remove selected rows by index
        uploaded_df = uploaded_df.drop(row_indices)
        
        # Reset index to ensure continuous row IDs
        uploaded_df = uploaded_df.reset_index(drop=True)
        
        after_count = len(uploaded_df)
        deleted_count = before_count - after_count
        
        return f"<p><b>Deleted {deleted_count} row(s) successfully.</b></p>" + dataframe_to_html_with_id(uploaded_df)
        
    except Exception as e:
        print(f"Error in bulk_delete_rows: {str(e)}")
        return f"Error deleting rows: {str(e)}", 500


@app.route('/reload_data')
def reload_data():
    global uploaded_df

    if uploaded_df.empty:
        return "No data to reload.", 400

    print(f"Reloading FULL dataset, total rows: {len(uploaded_df)}")

    # ✅ RETURN FULL DATASET — NOT SLICED
    return dataframe_to_html_with_id(uploaded_df)




@app.route('/update_cell', methods=['POST'])
def update_cell():
    global uploaded_df
    try:
        row_id = int(request.form.get('row_id'))  # This is now the actual DataFrame index
        column_name = request.form.get('column_name')
        new_value = request.form.get('new_value')
        original_value = request.form.get('original_value')
        
        if column_name not in uploaded_df.columns:
            return f"Column '{column_name}' not found", 400
        
        if row_id < 0 or row_id >= len(uploaded_df):
            return "Invalid row id", 400
        
        # Debug: Print the update info
        print(f"Updating cell: Row {row_id}, Column '{column_name}', From '{original_value}' To '{new_value}'")
        print(f"Current value before update: {uploaded_df.at[row_id, column_name]}")
        
        save_history()
        
        # Update the cell value
        uploaded_df.at[row_id, column_name] = new_value
        
        # Debug: Verify the update
        print(f"Current value after update: {uploaded_df.at[row_id, column_name]}")
        
        return "Cell updated successfully", 200
        
    except Exception as e:
        print(f"Error updating cell: {str(e)}")
        return f"Error updating cell: {str(e)}", 500


if __name__ == "__main__":
    if not os.path.exists('static'):
            os.makedirs('static')
    app.run(host="127.0.0.1", port=5001, debug=True)
