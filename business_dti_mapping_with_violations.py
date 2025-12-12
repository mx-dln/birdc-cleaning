import re
import math
import pandas as pd
from difflib import SequenceMatcher

INPUT_CSV = "Business Masterlist 2025.csv"
OUTPUT_CSV = "Business-2025-Mapped-SubCategories.csv"

# ---------------------------
# Load dataframe
# ---------------------------
df = pd.read_csv(INPUT_CSV, dtype=str)  # read everything as string to avoid dtype issues
df.fillna("", inplace=True)

# ---------------------------
# Detect LOB column (flexible)
# ---------------------------
lob_col = None
for c in df.columns:
    if "line" in c.lower() and "business" in c.lower():
        lob_col = c
        break
if lob_col is None:
    raise Exception("Missing 'Line of Business' column. Column name must contain 'line' and 'business'.")

# Make a working LOB_clean column
def normalize(text: str) -> str:
    s = (text or "")
    # unify separators, remove excessive punctuation, keep words
    s = s.replace("–", "-").replace("—", "-").replace("&", " and ")
    s = re.sub(r"[()\"'<>;:]", " ", s)
    s = re.sub(r"[/\\_\[\]\{\}]", " ", s)
    s = re.sub(r"[^0-9A-Za-z\-_\.&, ]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()

df["LOB_clean"] = df[lob_col].apply(normalize)

# ---------------------------
# DTI categories (canonical strings)
# ---------------------------
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

# For difflib fallback we want a flat list of all canonical strings
DTI_CANONICAL = list(DTI_CATEGORIES.values())

# ---------------------------
# RULES: pattern -> (category_key, priority)
# Lower priority number = higher precedence when multiple matches occur.
# Ordering and careful word-boundary usage prevents substring misfires (e.g., "art" in "apartment").
# ---------------------------
RULE_DEFS = [
    # priority 1: Manufacturing (food processors) - user confirmed mapping
    (r"\b(food processor|food products|food processing|meat processor|meat processing|frozen good|frozen goods|can[n]?ery|processing plant|packing plant|packer)\b", "Manufacturing Of Essential Goods", 1),

    # priority 2: Dining / Food preparation (take-out) and specific food stands
    (r"\b(burger stand|food stand|food stall|snack bar|snackbar|panciteria|kitchen|food services|refreshment|refreshments|refreshment stand|grill|grilled|bbq|barbecue)\b", "Food_Preparation_TakeOut", 2),

    # priority 3: Dining dine-in explicit
    (r"\b(restaurant|resto|eatery|kainan|dine in|dine-in|diner|vídeoke|videoke|karaoke.*bar|license to serve)\b", "Dining Dine In", 3),

    # priority 4: Essential retail specifics (sari-sari, rice, grocery, dried fish, vendors)
    (r"\b(sari[- ]?sari|grocery|groceries|grocer|rice retailer|rice retailer|rice seller|rice|corn retailer|corn|dried fish|dried-fish|driedfish|vegetable vendor|veg vendor|meat vendor|fish vendor|fish vendor|fish vendor|market|market stall|market vendor|vegetable vendor|balot|balut|palay|corn trading)\b", "Essential Retail", 4),

    # priority 5: Drugstore/pharmacy (explicit)
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
    (r"\b(vulcaniz|vulcanizing|vulcanize|auto repair|auto works|motor parts|motor parts|auto supply|battery repair|repair shop|car wash|auto shop|mechanic)\b", "AutoRepair", 10),

    # priority 11: Computer / internet shops / IT equipment
    (r"\b(computer shop|internet cafe|internet shop|it and communication|computer repair|computer retailer|computer dealer)\b", "Internet/ComputerShops", 11),

    # priority 12: Veterinary / pet related
    (r"\b(veterinary|vet clinic|animal clinic|pet shop|pet food|veterinary clinic)\b", "Veterinary Activities", 12),

    # priority 13: Hotel / boarding house / accommodation
    (r"\b(hotel|motel|boarding house|apartelle|resort|lodging|accommodation)\b", "Hotel/Accommodation", 13),

    # priority 14: Gym / fitness (ensure this sits above studio->media risk)
    (r"\b(gym|fitness|fitness studio|fitness center|sports facility|sports centre|sports center|gymnasium)\b", "Gyms/Fitness", 14),

    # priority 15: Salon / barber
    (r"\b(salon|barber|barbershop|parlor|parlour|beauty shop|beauty products)\b", "Salon/Barber", 15),

    # priority 16: Printing / publishing / office equipment
    (r"\b(print|printing|printing services|printing press|publisher|publishing)\b", "Publishing And Printing", 16),

    # priority 17: Bookstore / school & office supplies
    (r"\b(bookstore|school supplies|stationery|office supplies)\b", "Bookstore/OfficeSupplies", 17),

    # priority 18: Clothing / boutique / ukay
    (r"\b(boutique|clothes|clothing|apparel|ukay|garment|garments|tailor|dressmaker)\b", "Clothing And Accessories", 18),

    # priority 19: Toy / music / art (narrow patterns)
    (r"\b(toy store|toy\b|playground|amusement)\b", "Toy Store", 19),
    (r"\bmusic store\b|\bmusic\b", "Music Stores", 19),
    (r"\bart gallery\b|\bgallery\b", "Art Galleries", 19),

    # priority 20: Real estate / lessor / renting / leasing
    (r"\blessor\b|\bleasing\b|\brental\b|\brentals\b|\bapartment\b|\breal property lessor\b|\breal estate office\b", "RealEstate Leasing", 20),

    # priority 21: Funeral
    (r"\b(funeral|embalm|embalming)\b", "Funeral", 21),

    # priority 22: Delivery/logistics
    (r"\b(delivery|courier|logistics|freight|cargo|warehouse|warehousing|shipping|freight forward)\b", "Delivery/Logistics", 22),

    # priority 23: Generic "dealer/trading" fallback (we'll inspect tokens)
    (r"\b(dealer|dealer of|dealer/|trading|trader|trading company|wholesaler|wholesale|retailer|retail)\b", None, 23),

    # priority 99: Everything else -> Other Services (Junkshop is explicitly Other Services per your choice)
    (r".+", "Other Services", 99),
]

# ---------------------------
# Token sets for special logic / post-fix rules
# ---------------------------
ESSENTIAL_KEYWORDS = {
    "rice", "corn", "market", "grocery", "groceries", "sari-sari", "sari sari", "dried", "driedfish", "dried-fish",
    "vegetable", "veg", "meat", "fish", "balut", "balot", "palay", "bakery", "bakery goods", "bakery shop"
}
MANUFACTURING_KEYWORDS = {
    "food processor", "food processing", "processing", "packing", "packer", "meat processor", "frozen goods", "canning", "canner", "canning plant"
}
JUNK_KEYWORDS = {"junkshop", "junk shop", "scrap", "scrapyard", "waste dealer"}

# ---------------------------
# DTI Subcategory Violation Detection Rules
# ---------------------------
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

# ---------------------------
# Helper: sequence ratio difflib (safer fallback)
# ---------------------------
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

# ---------------------------
# Main mapping function (returns canonical string, method, score)
# - For multi-key "Line of Business" entries, extract tokens and choose the highest priority match.
# ---------------------------
def map_lob_to_dti(lob_text: str):
    text = (lob_text or "").strip().lower()
    if text == "":
        return DTI_CATEGORIES["Other Services"], "empty", 1.0

    # quick normalization for matching
    t = text

    # If the whole row mentions JUNKSHOP -> user wanted Other Services
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
                # if cat_key is None (dealer/trading generic) we will defer to token inspection
                matches.append((priority, cat_key, pattern))
        except re.error:
            continue

    # If we have concrete category matches (not None keys), pick the one with lowest priority value (highest precedence)
    concrete = [m for m in matches if m[1] is not None]
    if concrete:
        concrete.sort(key=lambda x: x[0])
        chosen = concrete[0]
        # chosen[1] is the key, map to canonical string if key is in DTI_CATEGORIES, else if it was already canonical return it
        chosen_key = chosen[1]
        if chosen_key in DTI_CATEGORIES:
            return DTI_CATEGORIES[chosen_key], f"rule:{chosen[2]}", 1.0
        else:
            # sometimes chosen_key is already canonical
            return str(chosen_key), f"rule:{chosen[2]}", 1.0

    # If only generic dealer/trading matched or no match, inspect tokens to decide
    tokens = set(re.findall(r"\b[\w\-]+\b", t))
    # If any essential keyword present, classify as Essential Retail
    if tokens & ESSENTIAL_KEYWORDS:
        return DTI_CATEGORIES["Essential Retail"], "token_essential_keyword", 1.0

    # Dealer/trader: try to detect what they deal in by looking for item tokens
    # Prioritize by obvious tokens
    if "dealer" in t or "trading" in t or "trader" in t or "wholesaler" in t or "wholesale" in t or "retailer" in t:
        # check for hardware terms
        hardware_terms = {"hardware", "lumber", "cement", "steel", "construction supply", "aggregates", "aggregate"}
        if any(h in t for h in hardware_terms):
            return DTI_CATEGORIES["Hardware Stores"], "dealer_item:hardware", 1.0
        # check for grocery/food
        if tokens & ESSENTIAL_KEYWORDS:
            return DTI_CATEGORIES["Essential Retail"], "dealer_item:essential", 1.0
        # check for auto parts
        if any(x in t for x in ["motor parts", "auto supply", "auto parts", "car parts"]):
            return DTI_CATEGORIES["AutoRepair"], "dealer_item:auto", 1.0
        # check for veterinary / agricultural
        if any(x in t for x in ["veterinary", "animal", "feed", "fertilizer"]):
            return DTI_CATEGORIES["Veterinary Activities"], "dealer_item:vet", 1.0
        # fallback for dealer: classify as Essential Retail (conservative)
        return DTI_CATEGORIES["Essential Retail"], "dealer_item:fallback_to_retail", 0.9

    # As a last resort use difflib semantic fallback but be cautious
    match, score = best_difflib_match(t, cutoff=0.60)
    if match:
        return match, "difflib", float(round(score, 3))

    # Final default: Other Services (this also covers Junkshop per choice)
    return DTI_CATEGORIES["Other Services"], "default_other", 0.0

# ---------------------------
# NEW: DTI Subcategory Violation Detection Function
# ---------------------------
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
# Apply mapping and produce match metadata
# ---------------------------
mapped = df["LOB_clean"].apply(lambda x: pd.Series(map_lob_to_dti(x), index=["DTI Sub Category", "Match_Method", "Match_Score"]))
df = pd.concat([df, mapped], axis=1)

# ---------------------------
# Apply violation detection
# ---------------------------
violation_results = df.apply(lambda row: pd.Series(detect_dti_violations(row["LOB_clean"], row["DTI Sub Category"]), 
                                                  index=["Violation_Status", "Violation_Details", "Violation_Count"]), axis=1)
df = pd.concat([df, violation_results], axis=1)

# ---------------------------
# Ensure required output columns exist; if missing in input create blank columns
# Requested table columns:
# Business Trade Name, Address, Barangay, Male personnel, Female personnel, DTI Sub Category, Line of Business, Year
# ---------------------------
requested_columns = [
    "Business Trade Name",
    "Address",
    "Barangay",
    "Male personnel",
    "Female personnel",
    "DTI Sub Category",
    lob_col,
    "Year"
]
# Create any missing with empty string to ensure CSV has consistent columns
for col in requested_columns:
    if col not in df.columns:
        df[col] = ""

# Reorder output columns to the requested order plus Match_Method/Score and violation info
output_cols = [
    "Business Trade Name",
    "Address",
    "Barangay",
    "Male personnel",
    "Female personnel",
    "DTI Sub Category",
    lob_col,
    "Year",
    "Match_Method",
    "Match_Score",
    "Violation_Status",
    "Violation_Details",
    "Violation_Count"
]

# Keep only existing ones in that order (defensive)
output_existing = [c for c in output_cols if c in df.columns]

# Save CSV
df.to_csv(OUTPUT_CSV, columns=output_existing, index=False)

print("✔ Done — output saved to:", OUTPUT_CSV)
print("Sample of mapped results with violation detection (first 20 rows):")
print(df[output_existing].head(20).to_string(index=False))

# Print violation summary
print("\n" + "="*50)
print("VIOLATION SUMMARY")
print("="*50)
violation_summary = df["Violation_Status"].value_counts()
print(violation_summary)
print(f"\nTotal businesses with potential violations: {len(df[df['Violation_Status'] == 'Potential Violation'])}")
print(f"Total compliant businesses: {len(df[df['Violation_Status'] == 'Compliant'])}")

# Show top violation types
if len(df[df['Violation_Status'] == 'Potential Violation']) > 0:
    print("\nTop 10 Most Common Violation Types:")
    all_violations = df[df['Violation_Status'] == 'Potential Violation']['Violation_Details'].str.split(' | ')
    violation_counts = {}
    for violations in all_violations:
        for violation in violations:
            violation_counts[violation] = violation_counts.get(violation, 0) + 1
    
    sorted_violations = sorted(violation_counts.items(), key=lambda x: x[1], reverse=True)
    for violation, count in sorted_violations[:10]:
        print(f"  {violation}: {count}")
