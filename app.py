import streamlit as st
from google import genai
from google.genai import types
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig, Part
import json
import base64
from PIL import Image
import io
import time
import requests
import zipfile
import fitz  # PyMuPDF
from json import JSONDecodeError
import os
from google.oauth2 import service_account
os.environ["GOOGLE_CLOUD_DISABLE_GRPC"] = "true"
PYMUPDF_AVAILABLE = True


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Analyzer",
    page_icon="🏠",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a3c5e 0%, #2d6a9f 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .main-header h1 { margin: 0; font-size: 1.8rem; }
    .main-header p  { margin: 0.3rem 0 0; opacity: 0.85; font-size: 0.95rem; }

    .result-card {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }
    .result-card h4 { margin: 0 0 0.5rem; color: #1a3c5e; font-size: 1rem; }

    .badge-pass {
        background: #d4edda; color: #155724;
        border: 1px solid #c3e6cb;
        border-radius: 6px; padding: 2px 10px; font-weight: 600; font-size: 0.82rem;
    }
    .badge-fail {
        background: #f8d7da; color: #721c24;
        border: 1px solid #f5c6cb;
        border-radius: 6px; padding: 2px 10px; font-weight: 600; font-size: 0.82rem;
    }
    .badge-warn {
        background: #fff3cd; color: #856404;
        border: 1px solid #ffeeba;
        border-radius: 6px; padding: 2px 10px; font-weight: 600; font-size: 0.82rem;
    }
    .badge-info {
        background: #cce5ff; color: #004085;
        border: 1px solid #b8daff;
        border-radius: 6px; padding: 2px 10px; font-size: 0.82rem;
    }

    .stExpander { border: 1px solid #dee2e6 !important; border-radius: 8px !important; }
    div[data-testid="stForm"] { border: none !important; }
    .upload-hint { color: #6c757d; font-size: 0.85rem; text-align: center; margin-top: 0.3rem; }
</style>
""", unsafe_allow_html=True)

# ?? Brochure USP Analyzer (merged from Previouscode.py) ????????????????????????

# ── Prompts ────────────────────────────────────────────────────────────────────

base_prompt = """You are an expert real estate copywriter specializing in premium residential properties. You have been provided a brochure PDF for a premium residential project.

Your task is to extract powerful, factual Unique Selling Propositions (USPs) that will compel high-net-worth buyers to take action.

═══════════════════════════════════════════
EXTRACTION SCOPE — cover ALL of these angles:
═══════════════════════════════════════════
1. Thematic & Architectural Identity — design philosophy, style, signature elements
2. Clubhouse & Lifestyle Amenities — name every facility with its size/count if stated
3. Technology, Automation & Security — smart home, surveillance, access control
4. Landscape, Green & Open Spaces — area %, acres, named gardens, water bodies
5. Location & Connectivity — distances to landmarks, roads, transit hubs (use exact km/min)
6. Developer, Architect & Consultant Pedigree — ONLY if a proper name is explicitly written
7. Awards, Certifications & Approvals — include certifying body and year if available
8. Unit & Project Specifications — total units, density, floors, BHK range, super area
9. Any other distinctive lifestyle, wellness, or convenience feature

═══════════════════════════════════════════
QUALITY RULES (non-negotiable):
═══════════════════════════════════════════
• FACTUAL PRECISION: Every USP must reflect data explicitly present in the brochure.
  - Prefer numbers over adjectives: "2-acre landscaped podium" beats "large garden"
  - Prefer named entities: "Palladian architecture" beats "unique design"
• PROPER NOUNS ONLY: Include architect/designer/consultant names ONLY if a proper name
  (not a generic title) is printed in the brochure.
• NO NOISE: Ignore boilerplate marketing copy, generic slogans, and legal disclaimers.
• NO HEADERS: Output is a flat list — no section titles or groupings.
• GRAMMAR: Capitalize proper nouns; use active, professional language.
• RANKING: Most unique and buyer-influential USP first; descending order throughout.

═══════════════════════════════════════════
CHARACTER LIMIT — STRICTLY ENFORCED:
═══════════════════════════════════════════
• Each USP text must be ≤ 75 characters (including spaces and punctuation).
• Count carefully. If a draft exceeds 75 characters:
    Step 1 — Remove filler words ("featuring", "offering", "boasting").
    Step 2 — Use numerals instead of words ("3" not "three").
    Step 3 — Use "&" instead of "and"; abbreviate units ("sq ft", "km", "min").
    Step 4 — Cut the least important qualifier.
  Never sacrifice a key fact to meet the limit — restructure instead.
• Do NOT truncate mid-word or mid-fact.

═══════════════════════════════════════════
CATEGORIZATION — follow exactly:
═══════════════════════════════════════════
Assign exactly ONE category from this fixed list:
  AMENITIES, LOCATION_AND_CONNECTIVITY, CONSTRUCTION_AND_DESIGN,
  TECHNOLOGY_AND_AUTOMATION, OFFERS, CERTIFICATES_AND_APPROVALS,
  AWARDS_AND_ACCOLADES, MASTER_PLAN

Sub-category rules:

▸ AMENITIES → pick exactly one from this fixed list only:
  ROOFTOP_LOUNGE, CHESS, VISITORS_PARKING, SAUNA, MULTIPURPOSE_COURT,
  ARTS_AND_CRAFTS_STUDIO, MUSIC_ROOM, THEME_PARK, AYURVEDIC_CENTRE,
  MASSAGE_ROOM, SALON, AIR_HOCKEY, GYMNASIUM, FOOSBALL, RESTAURANT,
  SWIMMING_POOL, CAFETERIA, LIBRARY, CARD_ROOM, CO_WORKING_SPACES,
  COMMUNITY_GARDEN_URBAN_FARMING, CLUB_HOUSE, BUSINESS_LOUNGE,
  CRICKET_PITCH, STEAM_ROOM, TOT_LOT, AMPHITHEATRE, ESCALATOR,
  REFLEXOLOGY_PARK, JOGGING_TRACK, CARROM, GREEN_WALL, WATER_PARK_SLIDES,
  INDOOR_GAMES, TABLE_TENNIS, FOOTBALL, SCHOOL, YOGA_MEDITATION_AREA,
  FOOD_COURT, BADMINTON_COURT, MEDICAL_CENTRE, CIGAR_LOUNGE, CLINIC,
  FLOWER_GARDEN, SQUASH_COURT, BILLIARDS, CAR_WASH_AREA, GAZEBO, PARKING,
  LANDSCAPE_GARDEN, TEMPLE, BARBEQUE, CYCLING_TRACK, CRECHE, LIFT,
  THEATER_HOME, SENIOR_CITIZEN_SITOUT, AEROBICS_CENTRE, AUTOMATED_CAR_WASH,
  BANQUET_HALL, SAND_PIT, PEDESTRIAN_FRIENDLY_ZONES, MULTIPURPOSE_HALL,
  EXTERIOR_LANDSCAPE, CAR_PARKING, GAMING_ZONES, PRIVATE_GARDENS_BALCONIES,
  MINI_THEATRE, GROCERY_SHOP, TERRACE_GARDEN, ARCHERY_RANGE, GOLF_COURSE,
  ATM, SKATING_RINK, BASKETBALL_COURT, NATURE_TRAIL, SHOPPING_CENTRE,
  PERGOLA, POOL_TABLE, PAVED_COMPOUND, LOUNGE, TODDLER_POOL,
  COMMUNITY_HALL, PARTY_LAWN, READING_LOUNGE, FOUNTAIN, JACUZZI,
  POWER_SUBSTATION, CENTRALIZED_AIR_CONDITIONING, SIT_OUT_AREA,
  CHILDRENS_PLAY_AREA, LAWN_TENNIS_COURT, SPA, BAR_CHILL_OUT_LOUNGE,
  INTERNAL_ROAD, THEATRE, BOWLING_ALLEY, MANICURED_GARDEN,
  ACUPRESSURE_PARK, CONFERENCE_ROOM, FOREST_TRAIL,
  BEACH_VOLLEY_BALL_COURT, INFINITY_POOL, ACCUPRESSURE_PARK, OPEN_SPACE,
  DANCE_STUDIO, SUN_DECK, NATURAL_POND, ROCK_CLIMBING_WALL, DART_BOARD,
  EV_CHARGING_STATIONS
  → If no suitable sub-category exists, do NOT use AMENITIES; do NOT create a custom one.

▸ LOCATION_AND_CONNECTIVITY → pick exactly one from this fixed list only:
  BUS, ELECTRICITY, BEACH, PETROL_PUMP, COLLEGE, AIRPORT, MARKETS, METRO,
  STADIUM, HOSPITALITY, GREEN_BELT, PARK, WATER, GOLF_COURSE, ATM,
  HERITAGE_PLACES, BANK, MULTI_LEVEL_PARKING, RAILWAY, AMUSEMENT_PARK,
  HIGHWAY, HOSPITAL, FLYOVER, MALLS, SCHOOL, MAJOR_ROAD, BUSINESS_HUB,
  PUBLIC_TRANSPORTATION
  → If no suitable sub-category exists, do NOT use LOCATION_AND_CONNECTIVITY; do NOT create a custom one.

▸ All other categories → create a custom sub-category in Title Case.
  It must NOT duplicate any predefined category or sub-category name.

═══════════════════════════════════════════
OUTPUT FORMAT — one line per USP, nothing else:
═══════════════════════════════════════════
[CATEGORY] | [SUB_CATEGORY] | [USP text ≤ 75 characters]

Examples of correct output:
AMENITIES | CLUB_HOUSE | 1 lakh sq ft clubhouse — largest in the micro-market
AMENITIES | SWIMMING_POOL | Infinity pool at 40th floor with panoramic city views
CONSTRUCTION_AND_DESIGN | Palladian Architecture | Neo-classical façade by award-winning Hafeez Contractor
LOCATION_AND_CONNECTIVITY | METRO | 5-min walk to Magenta Line metro station
MASTER_PLAN | Low Density | Only 400 units across 10 acres — 40 units/acre density
TECHNOLOGY_AND_AUTOMATION | Smart Home | Alexa-enabled smart home automation in every unit
CERTIFICATES_AND_APPROVALS | RERA Status | RERA registered; OC received — ready to move in
AWARDS_AND_ACCOLADES | Cnbc Awaaz Award | CNBC Awaaz Real Estate Award 2023 — Best Luxury Project

Begin extraction now. Output ONLY the formatted USP lines. No preamble, no summary, no extra text.
"""

old_usps_prompt = """
Additionally, I'm providing you with a list of previously identified USPs for this or a similar property. Review these old USPs alongside the brochure.

OLD USPs:
{old_usps}

Merge insights from both sources: remove duplicates, keep the most compelling and unique points from each. Apply the same formatting, character limit, and quality rules to all final USPs.
"""

specifications_prompt = """You are a real estate data extraction specialist. Extract all construction and material specifications from the provided document/source.

═══════════════════════════════════════════
WHAT TO EXTRACT:
═══════════════════════════════════════════
Extract every specification item you can find, including but not limited to:
Structure, Super Structure, Flooring, Plastering, Painting, Electrical, Main Door,
Internal Door, Windows, Kitchen, Toilets / Bathrooms, Water Proofing, Lift, Generator,
CCTV Cameras, Intercom, Security, Plumbing, Terrace, Common Areas, Lobby, Staircase,
External Façade, Compound Wall, Fire Fighting, STP / WTP, Rainwater Harvesting,
Solar, EV Charging, Parking, Gas Pipeline — and any other spec present in the source.

═══════════════════════════════════════════
QUALITY RULES:
═══════════════════════════════════════════
• Extract ONLY data explicitly stated in the source. Do NOT infer or assume.
• Use exact brand names, grades, and measurements as written.
• Preserve all sub-labels (e.g. "Internal:", "External:") within the description.
• If a spec is not mentioned, skip it — do NOT write "Not mentioned" or "N/A".
• The label should be a short, clean heading (e.g. "Structure", "Main Door", "Flooring").
• The description should be a single continuous paragraph capturing ALL detail for that item.
• SENTENCE CASE: Write every description in proper sentence case — first word capitalised, rest lowercase except for brand names, acronyms, and proper nouns (e.g. "Asian Paints", "RCC", "UPVC", "ISI", "BIS", "OHT").

═══════════════════════════════════════════
OUTPUT FORMAT — strict JSON only:
═══════════════════════════════════════════
Return a JSON array where each element is an object with exactly two keys:
  "label"       — short heading for the specification item
  "description" — full detail as a single descriptive paragraph

Example output:
[
  {
    "label": "Structure",
    "description": "Footings, columns, beams and slabs in RCC grade."
  },
  {
    "label": "Flooring",
    "description": "Coral/RAK/Cera/Johnson or equivalent reputed make double charged vitrified tiles with 4\\" skirting."
  },
  {
    "label": "Painting",
    "description": "Internal: Asian/JK/Latif/Equivalent reputed make with Asian easy clean weather coat exterior paints for external walls. Texture finish for elevation. External: Asian Enamel paints over two coats of enamel paint with grey luppum for internal doors and grills."
  },
  {
    "label": "Main Door",
    "description": "Best teak wood door frames and teak wood shutters aesthetically designed with fine finished melamine polishing and designer hardware of reputed make. Height of main door: 7'."
  }
]

Output ONLY the JSON array. No preamble, no explanation, no markdown fences.
"""

PREDEFINED_AMENITIES = [
    "Private Gardens/Balconies", "Swimming Pool", "Internal Street Lights", "Gated Community",
    "Anti-termite Treatment", "Earthquake Resistant", "Paved Compound", "Permeable Pavement",
    "Vastu Compliant", "Wheelchair Accessible", "Grade A Building", "Feng Shui", "Society Office",
    "Heli-Pad", "Solar Lighting", "Well-Maintained Internal Roads", "Energy Efficient Lightining",
    "Community Hall", "Solar Panel", "Temple", "School", "Pet Park", "Solar Water Heating",
    "Co-Working Spaces", "Library", "Carrom", "Thermal Insulation", "Creche/Day Care",
    "Outdoor Event Spaces", "Air Hockey", "Football Ground", "Table Tennis", "Volley Ball Court",
    "Pool Table", "Chess", "Dart Board", "Billiards", "Foosball", "Cricket Pitch", "Bowling Alley",
    "Lawn Tennis Court", "Basketball Court", "Rock Climbing Wall", "Badminton Court",
    "Beach Volley Ball Court", "Spa", "Jacuzzi", "Acupressure Park", "Skating Rink", "Squash Court",
    "Massage Room", "Yoga/Meditation Area", "Sauna", "Futsal", "Reflexology Park", "Aerobics Centre",
    "Video Gaming Room", "Ayurvedic Centre", "Doctor on Call", "Steam Room", "Flower Garden",
    "Terrace Garden", "Medical Centre", "Gymnasium", "Open Space", "Landscape Garden", "Fountain",
    "Clinic", "Pilates Studios", "Natural Pond", "Pedestrian-Friendly Zones", "Manicured Garden",
    "Senior Citizen Sitout", "Archery Range", "Water Park/Slides", "Sit Out Area",
    "Community Garden/Urban Farming", "Green Wall (Vertical Gardens)", "Forest Trail",
    "Cabana Sitting", "Park", "Car Parking", "Art and Craft Studio", "EV Charging Stations",
    "Music Room", "Dance Studio", "Barbecue", "Banquet Hall", "Sun Deck", "Party Lawn", "Sand Pit",
    "Mini Theatre", "Club House", "Children's Play Area", "Multipurpose Hall", "Gazebo",
    "Amphitheatre", "Card Room", "Jogging Track", "Multipurpose court", "Theatre", "Golf Course",
    "Tot Lot", "Nature Trail", "Theater Home", "Cycling Track", "Art Gallery", "Fire Alarm",
    "Gaming Zones", "Boom Barrier", "Wine Cellar", "Emergency Exits", "Golf Simulator",
    "CCTV Camera Security", "Golf Putty", "Fire Fighting Systems", "Security Cabin", "Indoor Games",
    "Gas Leak Detectors", "Biometric/Smart Card Access", "Fire NOC", "Video Door Security",
    "Theme Park", "Smoke Detectors", "24x7 Security", "Panic Buttons in Apartments",
    "Rooftop Lounge", "Car-Free Zones", "Ambulance Service", "Cigar Lounge", "Intercom Facilities",
    "Emergency Evacuation Chairs", "Signage and Road Markings", "Lounge", "Bar/Chill-Out Lounge",
    "Fall Detection Systems in Bathrooms", "Defibrillators in Common Areas", "Piped Gas",
    "Business Lounge", "Restaurant", "Waiting Lounge", "Reading Lounge", "Wi-Fi Connectivity",
    "Pergola", "Smart Home Automation", "DTH Television", "Laundry", "Conference Room",
    "Wi-Fi Zones in Common Areas", "Cafeteria", "RO System", "Food Court", "Laundromat",
    "Shopping Centre", "Property Staff", "Changing Area", "Lifts", "Name Plates",
    "Automated Car Wash", "Concierge Service", "Toilet for Drivers", "Car Wash Area", "Salon",
    "Grocery Shop", "Bus Shelter", "Milk Booth", "Letter Box", "Petrol Pump", "Entrance Lobby",
    "24/7 Power Backup", "Maintenance Staff", "Intercom", "ATM", "DG Availability",
    "Power Back up Lift", "Escalators", "Noise Insulation in Apartments",
    "Centralized Air Conditioning", "Plumber/Electrician on Call", "Secretarial Services",
    "Underground Electric Cabling", "Power Substation", "Braille Signage", "Air Purification Systems",
    "Composting Facilities", "Recycling Facilities", "Garbage Chute", "Garbage Disposal",
    "Organic Waste Converter", "Waste Segregation and Disposal", "Waste Management",
    "Sewage Treatment Plant", "Water Treatment Plant", "Water Softener Plant", "Smart Water Meters",
    "Rain Water Harvesting", "Bioswales", "Ground Water Recharging Systems", "24/7 Water Supply",
    "Municipal Water Supply", "Low Flow Fixtures", "Greywater Recycling", "Borewell Water Supply",
]

amenities_extraction_prompt = """You are an information extraction assistant.

Your task is to extract amenities mentioned in the provided real estate brochure PDF.

IMPORTANT RULES:
1. Only extract amenities that EXACTLY match items from the predefined amenities list provided below.
2. Do NOT add new amenities.
3. Do NOT infer or assume amenities.
4. If a brochure mentions something similar but not exactly matching the list, ignore it.
5. Return the output as a JSON array and nothing else — no preamble, no explanation, no markdown fences.
6. If no amenities from the list are found, return an empty array [].
7. Each amenity should appear only once in the output.
8. Preserve the exact spelling and format from the predefined list.

PREDEFINED AMENITIES LIST:
Private Gardens/Balconies, Swimming Pool, Internal Street Lights, Gated Community,
Anti-termite Treatment, Earthquake Resistant, Paved Compound, Permeable Pavement,
Vastu Compliant, Wheelchair Accessible, Grade A Building, Feng Shui, Society Office,
Heli-Pad, Solar Lighting, Well-Maintained Internal Roads, Energy Efficient Lightining,
Community Hall, Solar Panel, Temple, School, Pet Park, Solar Water Heating,
Co-Working Spaces, Library, Carrom, Thermal Insulation, Creche/Day Care,
Outdoor Event Spaces, Air Hockey, Football Ground, Table Tennis, Volley Ball Court,
Pool Table, Chess, Dart Board, Billiards, Foosball, Cricket Pitch, Bowling Alley,
Lawn Tennis Court, Basketball Court, Rock Climbing Wall, Badminton Court,
Beach Volley Ball Court, Spa, Jacuzzi, Acupressure Park, Skating Rink, Squash Court,
Massage Room, Yoga/Meditation Area, Sauna, Futsal, Reflexology Park, Aerobics Centre,
Video Gaming Room, Ayurvedic Centre, Doctor on Call, Steam Room, Flower Garden,
Terrace Garden, Medical Centre, Gymnasium, Open Space, Landscape Garden, Fountain,
Clinic, Pilates Studios, Natural Pond, Pedestrian-Friendly Zones, Manicured Garden,
Senior Citizen Sitout, Archery Range, Water Park/Slides, Sit Out Area,
Community Garden/Urban Farming, Green Wall (Vertical Gardens), Forest Trail,
Cabana Sitting, Park, Car Parking, Art and Craft Studio, EV Charging Stations,
Music Room, Dance Studio, Barbecue, Banquet Hall, Sun Deck, Party Lawn, Sand Pit,
Mini Theatre, Club House, Children's Play Area, Multipurpose Hall, Gazebo,
Amphitheatre, Card Room, Jogging Track, Multipurpose court, Theatre, Golf Course,
Tot Lot, Nature Trail, Theater Home, Cycling Track, Art Gallery, Fire Alarm,
Gaming Zones, Boom Barrier, Wine Cellar, Emergency Exits, Golf Simulator,
CCTV Camera Security, Golf Putty, Fire Fighting Systems, Security Cabin, Indoor Games,
Gas Leak Detectors, Biometric/Smart Card Access, Fire NOC, Video Door Security,
Theme Park, Smoke Detectors, 24x7 Security, Panic Buttons in Apartments,
Rooftop Lounge, Car-Free Zones, Ambulance Service, Cigar Lounge, Intercom Facilities,
Emergency Evacuation Chairs, Signage and Road Markings, Lounge, Bar/Chill-Out Lounge,
Fall Detection Systems in Bathrooms, Defibrillators in Common Areas, Piped Gas,
Business Lounge, Restaurant, Waiting Lounge, Reading Lounge, Wi-Fi Connectivity,
Pergola, Smart Home Automation, DTH Television, Laundry, Conference Room,
Wi-Fi Zones in Common Areas, Cafeteria, RO System, Food Court, Laundromat,
Shopping Centre, Property Staff, Changing Area, Lifts, Name Plates,
Automated Car Wash, Concierge Service, Toilet for Drivers, Car Wash Area, Salon,
Grocery Shop, Bus Shelter, Milk Booth, Letter Box, Petrol Pump, Entrance Lobby,
24/7 Power Backup, Maintenance Staff, Intercom, ATM, DG Availability,
Power Back up Lift, Escalators, Noise Insulation in Apartments,
Centralized Air Conditioning, Plumber/Electrician on Call, Secretarial Services,
Underground Electric Cabling, Power Substation, Braille Signage, Air Purification Systems,
Composting Facilities, Recycling Facilities, Garbage Chute, Garbage Disposal,
Organic Waste Converter, Waste Segregation and Disposal, Waste Management,
Sewage Treatment Plant, Water Treatment Plant, Water Softener Plant, Smart Water Meters,
Rain Water Harvesting, Bioswales, Ground Water Recharging Systems, 24/7 Water Supply,
Municipal Water Supply, Low Flow Fixtures, Greywater Recycling, Borewell Water Supply

Output ONLY a valid JSON array. No preamble, no explanation, no markdown code fences.
Example: ["Swimming Pool", "Gymnasium", "Club House"]
"""
project_id = st.secrets["GCP_PROJECT_ID"]
location = st.secrets["GCP_LOCATION"]
# ── Helper functions ───────────────────────────────────────────────────────────
@st.cache_resource
def setup_gemini_client():
    try:
        credentials = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=["https://www.googleapis.com/auth/cloud-platform"],  # ← add this
        )

        client = genai.Client(
            vertexai=True,
            project=st.secrets["GCP_PROJECT_ID"],
            location=st.secrets["GCP_LOCATION"],
            credentials=credentials,
        )

        return client

    except Exception as e:
        st.error(f"Gemini client error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None

def download_pdf_from_url(url):
    try:
        response = requests.get(url, stream=True, timeout=20)
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "")
        if "application/pdf" not in content_type and not url.lower().endswith(".pdf"):
            st.error(f"URL does not point to a valid PDF. Content-Type: {content_type}")
            return None
        return response.content
    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading PDF: {str(e)}")
        return None


def analyze_website_via_gemini(url, prompt, model_name, client):
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()

        html = response.text[:200000]

        with st.spinner(f"Analyzing website with {model_name}..."):

            result = client.models.generate_content(
                model=model_name,
                contents=[
                    prompt,
                    f"Website URL: {url}\n\nWebsite HTML:\n{html}",
                ],
            )

        return result.text

    except Exception as e:
        st.error(f"Website analysis error: {str(e)}")

        import traceback
        st.code(traceback.format_exc())

        return None


def analyze_pdf_via_files_api(pdf_bytes, prompt, model_name, client):
    try:
        with st.spinner(f"Analyzing with {model_name}..."):

            response = client.models.generate_content(
                model=model_name,
                contents=[
                    prompt,
                    types.Part.from_bytes(
                        data=pdf_bytes,
                        mime_type="application/pdf",
                    ),
                ],
            )

        return response.text

    except Exception as e:
        st.error(f"PDF analysis error: {str(e)}")

        import traceback
        st.code(traceback.format_exc())

        return None

    except Exception as e:
        st.error(f"Error during Vertex analysis: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None


import re

# Words/patterns that must stay uppercase regardless of position
_PRESERVE_UPPER = {
    "RCC", "PVC", "UPVC", "GI", "MS", "SS", "ISI", "BIS", "OHT", "STP", "WTP",
    "ETP", "MLD", "KVA", "KW", "AC", "DC", "MCB", "ELCB", "RCCB", "DB",
    "CCTV", "DTH", "RO", "UV", "TV", "LED", "CFL", "AHU", "VRF", "VRV",
    "RERA", "OC", "CC", "NOC", "BHK", "FSI", "FAR", "TDR",
    "CP", "WPC", "HDF", "MDF", "BWR", "BWP",
}

def _sentence_case(text: str) -> str:
    if not text:
        return text

    def fix_sentence(sentence: str) -> str:
        sentence = sentence.strip()
        if not sentence:
            return sentence
        words = sentence.split()
        result = []
        for i, word in enumerate(words):
            stripped = word.rstrip(".,;:\"')")
            leading = word[: len(word) - len(word.lstrip("(\"'"))]
            trailing = word[len(leading) + len(stripped):]

            if "/" in stripped:
                parts = stripped.split("/")
                fixed_parts = []
                for p in parts:
                    p_upper = p.upper()
                    if p_upper in _PRESERVE_UPPER:
                        fixed_parts.append(p_upper)
                    elif p and p[0].isupper():
                        fixed_parts.append(p)
                    else:
                        fixed_parts.append(p.lower())
                result.append(leading + "/".join(fixed_parts) + trailing)
                continue

            upper_stripped = stripped.upper()
            if upper_stripped in _PRESERVE_UPPER:
                result.append(leading + upper_stripped + trailing)
            elif i == 0:
                result.append(leading + stripped[:1].upper() + stripped[1:].lower() + trailing)
            elif stripped and stripped[0].isupper() and len(stripped) > 1 and stripped[1].islower():
                result.append(word)
            else:
                result.append(leading + stripped.lower() + trailing)

        return " ".join(result)

    sentences = re.split(r'(?<=[.!?])\s+', text)
    return "  ".join(fix_sentence(s) for s in sentences)


def apply_sentence_case(specs_data: list) -> list:
    result = []
    for item in specs_data:
        result.append({
            "label": item.get("label", "").strip(),
            "description": _sentence_case(item.get("description", "").strip()),
        })
    return result


def _is_blank_or_mask(img_bytes: bytes, ext: str, blank_threshold: float = 0.97) -> bool:
    try:
        from PIL import Image as PILImage
        import io as _io

        pil_img = PILImage.open(_io.BytesIO(img_bytes)).convert("RGB")
        pixels = list(pil_img.getdata())
        total = len(pixels)
        if total == 0:
            return True

        from collections import Counter
        sample = pixels[::8]
        if not sample:
            return True
        most_common_colour, _ = Counter(sample).most_common(1)[0]
        close = sum(
            1 for p in sample
            if abs(p[0] - most_common_colour[0]) <= 10
            and abs(p[1] - most_common_colour[1]) <= 10
            and abs(p[2] - most_common_colour[2]) <= 10
        )
        if close / len(sample) >= blank_threshold:
            return True

        bw_count = sum(
            1 for p in sample
            if (p[0] < 10 and p[1] < 10 and p[2] < 10)
            or (p[0] > 245 and p[1] > 245 and p[2] > 245)
        )
        if bw_count / len(sample) >= 0.97:
            return True

        return False

    except Exception:
        return False


def extract_images_from_pdf(pdf_bytes, min_width=200, min_height=200):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    smask_xrefs = set()
    for page in doc:
        for img_info in page.get_images(full=True):
            smask = img_info[8]
            if smask:
                smask_xrefs.add(smask)

    seen_xrefs = set()
    images = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        img_list = page.get_images(full=True)

        for img_index, img_info in enumerate(img_list):
            xref = img_info[0]

            if xref in smask_xrefs or xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)

            try:
                base_image = doc.extract_image(xref)
                width  = base_image["width"]
                height = base_image["height"]

                if width < min_width or height < min_height:
                    continue

                ext       = base_image["ext"]
                img_bytes = base_image["image"]

                cs = base_image.get("colorspace", 3)
                if cs not in (1, 3):
                    pix = fitz.Pixmap(doc, xref)
                    if pix.n > 4:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    img_bytes = pix.tobytes("png")
                    ext = "png"

                if _is_blank_or_mask(img_bytes, ext):
                    continue

                images.append({
                    "name":   f"page{page_num + 1}_img{img_index + 1}.{ext}",
                    "data":   img_bytes,
                    "ext":    ext,
                    "width":  width,
                    "height": height,
                    "page":   page_num + 1,
                })

            except Exception:
                continue

    doc.close()
    return images


def render_specifications(specs_data):
    if not specs_data:
        st.info("No specifications could be extracted.")
        return

    for item in specs_data:
        label = item.get("label", "").strip()
        description = item.get("description", "").strip()
        if label and description:
            st.markdown(f"**{label}**")
            st.write(description)
            st.divider()



def render_brochure_usp_analyzer():
    st.title("USP using Gemini")

    if "extracted_images" not in st.session_state:
        st.session_state.extracted_images = None

    # ── UI ─────────────────────────────────────────────────────────────────────────

    st.write("Upload Brochure, Enter Brochure URL, or Enter Project Website Link — then optionally add Old USPs")

    # Model selection
    st.subheader("Select Gemini Model")
    model_options = {
        "Gemini 3.1 pro": "gemini-3.5-flash",
        "Gemini 2.5 flash": "gemini-2.5-flash",
    }
    selected_model_name = st.selectbox(
        "Choose the AI model for analysis / Switch models if facing errors",
        options=list(model_options.keys()),
        index=0,
    )
    selected_model = model_options[selected_model_name]

    # ── Input sources ──────────────────────────────────────────────────────────────
    st.subheader("Input Source")

    col_upload, col_pdfurl, col_weburl = st.columns(3)

    with col_upload:
        st.markdown("**📄 Upload Brochure PDF**")
        uploaded_file = st.file_uploader("Upload a brochure file", type=["pdf"], label_visibility="collapsed")
        MAX_FILE_SIZE_MB = 20

        if uploaded_file is not None:
            file_size_mb = uploaded_file.size / (1024 * 1024)

            if file_size_mb > MAX_FILE_SIZE_MB:
                st.error(f"File exceeds {MAX_FILE_SIZE_MB} MB limit")
                st.stop()
    with col_pdfurl:
        st.markdown("**🔗 Brochure PDF URL**")
        pdf_url = st.text_input(
            "Enter URL to PDF brochure",
            placeholder="https://example.com/brochure.pdf",
            label_visibility="collapsed",
        )

    with col_weburl:
        st.markdown("**🌐 Project Website URL**")
        website_url = st.text_input(
            "Enter project website link",
            placeholder="https://projectname.com",
            label_visibility="collapsed",
        )

    # Old USPs
    st.subheader("Enter Old USPs (Optional)")
    old_usps = st.text_area("Paste previous USPs here", height=200)

    # ── Resolve input source ───────────────────────────────────────────────────────

    pdf_bytes = None
    input_source = None
    input_mode = None  # "pdf" or "website"

    if uploaded_file is not None:
        pdf_bytes = uploaded_file.read()
        uploaded_file.seek(0)
        input_source = f"File: {uploaded_file.name}"
        input_mode = "pdf"
    elif pdf_url and pdf_url.strip():
        pdf_bytes = download_pdf_from_url(pdf_url.strip())
        if pdf_bytes:
            input_source = f"PDF URL: {pdf_url}"
            input_mode = "pdf"
    elif website_url and website_url.strip():
        input_source = f"Website: {website_url}"
        input_mode = "website"

    # ── Main panels ────────────────────────────────────────────────────────────────
    st.subheader("Extraction Options")

    if input_mode:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Input Source")
            st.write(input_source)
            if input_mode == "pdf" and pdf_bytes:
                st.info(f"PDF size: {len(pdf_bytes) / 1024:.1f} KB")
            elif input_mode == "website":
                st.info("Website content will be scraped on extraction.")

        # ── Tab layout ─────────────────────────────────────────────────────────────
        tab_usp, tab_amenities, tab_specs, tab_images = st.tabs([
            "📋 Extract USPs",
            "🏊 Extract Amenities",
            "📐 Extract Specifications",
            "🖼️ Extract Images",
        ])

        # ── USP Tab ────────────────────────────────────────────────────────────────
        with tab_usp:
            st.info(f"Model: {selected_model_name}")
            analyze_button = st.button("Extract USPs", key="btn_usp")
            result_placeholder = st.empty()

            if analyze_button:
                client = setup_gemini_client()
                if not client:
                    st.stop()

                start_time = time.time()
                result_placeholder.info("Analyzing… this may take 30–90 seconds.")
                analysis = None

                if input_mode == "pdf":
                    full_prompt = base_prompt
                    if old_usps.strip():
                        full_prompt += old_usps_prompt.format(old_usps=old_usps)
                    analysis = analyze_pdf_via_files_api(pdf_bytes, full_prompt, selected_model, client)

                elif input_mode == "website":
                    full_prompt = base_prompt
                    if old_usps.strip():
                        full_prompt += old_usps_prompt.format(old_usps=old_usps)
                    analysis = analyze_website_via_gemini(
                        website_url.strip(),
                        full_prompt,
                        selected_model,
                        client
                    )

                execution_time = time.time() - start_time

                if analysis:
                    result_placeholder.empty()
                    usps = [u for u in analysis.strip().split("\n") if u.strip()]

                    st.subheader("Extracted USPs")

                    for usp in usps:
                        parts = usp.split("|")
                        if len(parts) == 3:
                            category = parts[0].strip()
                            subcategory = parts[1].strip()
                            text = parts[2].strip()
                            char_count = len(text)
                            over = char_count > 75

                            col_a, col_b = st.columns([1, 3])
                            with col_a:
                                st.markdown(f"**{category}**")
                                st.caption(subcategory)
                            with col_b:
                                st.markdown(f"• {text}")
                                if over:
                                    st.warning(f"⚠️ {char_count} chars (limit: 75)")
                            st.divider()
                        else:
                            st.markdown(f"• {usp}")

                    st.caption(
                        f"Analysis completed in {execution_time:.1f}s using {selected_model_name}. "
                        + ("PDF was analyzed with Vertex Gemini." if input_mode == "pdf" else "Website content was analyzed with Vertex Gemini.")
                    )

                    st.download_button(
                        label="Download USPs",
                        data=analysis,
                        file_name="property_usps.txt",
                        mime="text/plain",
                    )
                else:
                    result_placeholder.error("Failed to generate analysis. Please try again.")

        # ── Amenities Tab ──────────────────────────────────────────────────────────
        with tab_amenities:
            st.info(f"Model: {selected_model_name}")
            st.write(
                "Extracts only amenities that exactly match the predefined list "
                f"({len(PREDEFINED_AMENITIES)} items)."
            )
            amenities_button = st.button("Extract Amenities", key="btn_amenities")
            amenities_placeholder = st.empty()

            if amenities_button:
                client = setup_gemini_client()
                if not client:
                    st.stop()

                start_time = time.time()
                amenities_placeholder.info("Extracting amenities… this may take 30–90 seconds.")
                raw = None

                if input_mode == "pdf":
                    raw = analyze_pdf_via_files_api(
                        pdf_bytes,
                        amenities_extraction_prompt,
                        selected_model,
                        client
                    )

                elif input_mode == "website":
                    raw = analyze_website_via_gemini(
                        website_url.strip(),
                        amenities_extraction_prompt,
                        selected_model,
                        client
                    )

                execution_time = time.time() - start_time

                if raw:
                    amenities_placeholder.empty()

                    try:
                        clean = raw.strip().strip("```json").strip("```").strip()
                        extracted = json.loads(clean)
                    except json.JSONDecodeError:
                        st.error("Model returned unexpected format. Raw response shown below.")
                        st.code(raw)
                        extracted = []

                    if extracted:
                        valid = [a for a in extracted if a in PREDEFINED_AMENITIES]
                        invalid = [a for a in extracted if a not in PREDEFINED_AMENITIES]

                        st.subheader(f"Amenities Found ({len(valid)})")
                        cols = st.columns(3)
                        for i, amenity in enumerate(valid):
                            with cols[i % 3]:
                                st.markdown(f"✅ {amenity}")

                        if invalid:
                            st.warning(
                                f"⚠️ {len(invalid)} item(s) returned by the model did not match "
                                "the predefined list and were excluded:"
                            )
                            for item in invalid:
                                st.caption(f"  • {item}")

                        st.caption(f"Extraction completed in {execution_time:.1f}s using {selected_model_name}.")

                        st.download_button(
                            label="Download Amenities (JSON)",
                            data=json.dumps(valid, indent=2),
                            file_name="property_amenities.json",
                            mime="application/json",
                        )
                        st.download_button(
                            label="Download Amenities (TXT)",
                            data="\n".join(valid),
                            file_name="property_amenities.txt",
                            mime="text/plain",
                        )
                    else:
                        st.info("No matching amenities were found.")
                        st.caption(f"Extraction completed in {execution_time:.1f}s using {selected_model_name}.")
                else:
                    amenities_placeholder.error("Failed to extract amenities. Please try again.")

        # ── Specifications Tab ─────────────────────────────────────────────────────
        with tab_specs:
            st.info(f"Model: {selected_model_name}")
            st.write(
                "Extracts structured project specs: configuration details, construction materials, "
                "common area info, and more."
            )
            specs_button = st.button("Extract Specifications", key="btn_specs")
            specs_placeholder = st.empty()

            if specs_button:
                client = setup_gemini_client()
                if not client:
                    st.stop()

                start_time = time.time()
                specs_placeholder.info("Extracting specifications… this may take 30–90 seconds.")
                raw = None

                if input_mode == "pdf":
                    raw = analyze_pdf_via_files_api(
                        pdf_bytes,
                        specifications_prompt,
                        selected_model,
                        client
                    )
                
                elif input_mode == "website":
                    raw = analyze_website_via_gemini(
                        website_url.strip(),
                        specifications_prompt,
                        selected_model,
                        client
                    )

                execution_time = time.time() - start_time

                if raw:
                    specs_placeholder.empty()

                    try:
                        clean = raw.strip().strip("```json").strip("```").strip()
                        specs_data = json.loads(clean)
                    except json.JSONDecodeError:
                        st.error("Model returned unexpected format. Raw response shown below.")
                        st.code(raw)
                        specs_data = None

                    if specs_data:
                        specs_data = apply_sentence_case(specs_data)
                        st.subheader(f"Specifications ({len(specs_data)} items)")
                        render_specifications(specs_data)

                        st.caption(f"Extraction completed in {execution_time:.1f}s using {selected_model_name}.")

                        txt_output = "\n\n".join(
                            f"{item['label']}\n{item['description']}"
                            for item in specs_data
                            if item.get("label") and item.get("description")
                        )

                        col_dl1, col_dl2 = st.columns(2)
                        with col_dl1:
                            st.download_button(
                                label="Download Specifications (JSON)",
                                data=json.dumps(specs_data, indent=2),
                                file_name="property_specifications.json",
                                mime="application/json",
                            )
                        with col_dl2:
                            st.download_button(
                                label="Download Specifications (TXT)",
                                data=txt_output,
                                file_name="property_specifications.txt",
                                mime="text/plain",
                            )
                    else:
                        st.info("No specifications could be extracted.")
                        st.caption(f"Extraction completed in {execution_time:.1f}s using {selected_model_name}.")
                else:
                    specs_placeholder.error("Failed to extract specifications. Please try again.")

        # ── Images Tab ─────────────────────────────────────────────────────────────
        with tab_images:
            if input_mode == "website":
                st.warning("Image extraction is only available for PDF input (uploaded file or PDF URL). Please provide a brochure PDF.")
            elif not PYMUPDF_AVAILABLE:
                st.error("PyMuPDF is not installed. Add `pymupdf` to your requirements.txt to enable image extraction.")
            else:
                st.write("Extracts all images from the brochure PDF. Tiny icons and decorative elements are filtered out automatically.")

                col_minw, col_minh = st.columns(2)
                with col_minw:
                    min_width = st.slider("Minimum image width (px)", 50, 500, 200, step=50)
                with col_minh:
                    min_height = st.slider("Minimum image height (px)", 50, 500, 200, step=50)

                images_button = st.button("Extract Images", key="btn_images")
                images_placeholder = st.empty()

                # ✅ Run extraction and save to session state
                if images_button:
                    start_time = time.time()
                    images_placeholder.info("Extracting images from PDF…")

                    try:
                        st.session_state.extracted_images = extract_images_from_pdf(
                            pdf_bytes, min_width=min_width, min_height=min_height
                        )
                        execution_time = time.time() - start_time
                        images_placeholder.empty()
                        st.session_state["images_execution_time"] = execution_time

                    except Exception as e:
                        images_placeholder.error(f"Error extracting images: {str(e)}")
                        import traceback
                        st.error(traceback.format_exc())
                        st.session_state.extracted_images = None

                # ✅ Render from session state — persists across reruns triggered by download clicks
                if st.session_state.extracted_images:
                    extracted_images = st.session_state.extracted_images
                    execution_time = st.session_state.get("images_execution_time", 0)

                    st.subheader(f"Images Found ({len(extracted_images)})")

                    cols = st.columns(3)
                    for i, img in enumerate(extracted_images):
                        with cols[i % 3]:
                            mime = "image/png" if img["ext"] == "png" else f"image/{img['ext']}"
                            st.image(
                                img["data"],
                                caption=f"Page {img['page']} — {img['width']}×{img['height']}px",
                                use_container_width=True,
                            )
                            st.download_button(
                                label="⬇ Download",
                                data=img["data"],
                                file_name=img["name"],
                                mime=mime,
                                key=f"dl_img_{i}",
                            )

                    st.divider()

                    # Zip download for all images
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                        for img in extracted_images:
                            zf.writestr(img["name"], img["data"])
                    zip_buffer.seek(0)

                    st.download_button(
                        label=f"⬇ Download All Images as ZIP ({len(extracted_images)} files)",
                        data=zip_buffer,
                        file_name="brochure_images.zip",
                        mime="application/zip",
                    )

                    st.caption(f"Extraction completed in {execution_time:.1f}s.")

                elif st.session_state.extracted_images is not None and len(st.session_state.extracted_images) == 0:
                    st.info(
                        f"No images found meeting the minimum size. "
                        "Try lowering the size thresholds."
                    )
    else:
        st.caption("Choose an input source above to enable extraction.")
        tab_usp, tab_amenities, tab_specs, tab_images = st.tabs([
            "📋 Extract USPs",
            "🏊 Extract Amenities",
            "📐 Extract Specifications",
            "🖼️ Extract Images",
        ])

        with tab_usp:
            st.info("Upload a brochure PDF, enter a PDF URL, or enter a project website URL to extract USPs.")
        with tab_amenities:
            st.info("Upload a brochure PDF, enter a PDF URL, or enter a project website URL to extract amenities.")
        with tab_specs:
            st.info("Upload a brochure PDF, enter a PDF URL, or enter a project website URL to extract specifications.")
        with tab_images:
            st.info("Upload a brochure PDF or enter a PDF URL to extract images.")

    # Footer
    st.divider()
    st.caption("Premium Property USP Analyzer — Powered by Google Gemini")

tool_mode = st.sidebar.radio(
    "Tool",
    ["Consent Form Analyzer", "Brochure USP Analyzer"],
)


if tool_mode == "Brochure USP Analyzer":
    render_brochure_usp_analyzer()
    st.stop()

# ?? Header ????????????????????????????????????????????????????????????????????
st.markdown("""
<div class="main-header">
    <h1>🏠 Consent Form Analyzer</h1>
    <p>Upload a real estate consent form (image or PDF) — powered by Gemini Vision AI</p>
</div>
""", unsafe_allow_html=True)

# ── Schema ────────────────────────────────────────────────────────────────────
SCHEMA_DICT = {
  "type": "object",
  "description": "The input image may contain multiple pages placed side-by-side in a single image. Carefully scan the entire image from LEFT to RIGHT and TOP to BOTTOM. Treat each visible section as a continuation of the same document. Extract information from ALL visible pages, not just the first page. Do not stop reading after the first page. Information required for different fields may appear on any page.",
  "properties": {
    "project_name": {
      "type": "string",
      "description": "Name of the real estate project mentioned in the consent form."
    },
    "project_id": {
      "type": "string",
      "description": "Id related to a project, Also knows as XID. This can be found in a url or directly in mail text. Sample url (https://www.99acres.com/prime-alpha-apartments-ajmeri-gate-central-delhi-npxid-c431072). Here project_id is - 431072"
    },
    "res_com": {
      "type": "string",
      "enum": [
        "R",
        "C"
      ],
      "description": "Denotes whether a project is residential (R) or commercial (C). This can be found in a url or directly in mail text. Sample url (https://www.99acres.com/prime-alpha-apartments-ajmeri-gate-central-delhi-npxid-c431072). Here res_com is - C (commercial)"
    },
    "date": {
      "type": "string",
      "description": "Date mentioned on the consent form (DD/MM/YYYY) before To, heading. If unclear or not found, return null."
    },
    "address": {
      "type": "object",
      "properties": {
        "full_address": {
          "type": "string",
          "description": "Return the exact address mentioned in the form in the address placeholder. If no address is mentioned or the field is blank, return 'null'."
        },
        "locality": {
          "type": "string",
          "description": "Return Exact Locality/area of the project not the full address. If no locality is mentioned or cannot be determined, return 'null'."
        },
        "city": {
          "type": "string",
          "description": "Return City of the project. If no city is mentioned or cannot be determined, return 'null'."
        }
      },
      "required": [
        "locality",
        "city"
      ]
    },
    "possession": {
      "type": "object",
      "properties": {
        "date": {
          "type": "string",
          "description": "Possession date of the project. Extract the date from the form and return it strictly in YYYY-MM-DD format. Conversion rules: - If a full date is given (day, month, year), convert directly to YYYY-MM-DD. - If only month and year are given (e.g., 'Jan 2028'), use the LAST day of that month as the day (e.g., 'Jan 2028' → '2028-01-31', 'Feb 2028' → '2028-02-29' accounting for leap year, 'Mar 2028' → '2028-03-31'). - If only a year is given, use December 31 of that year (e.g., '2028' → '2028-12-31'). - Always return in YYYY-MM-DD format regardless of how the date is written in the form. - Return null if no possession date is mentioned anywhere in the form."
        },
        "status": {
          "type": "array",
          "items": {
            "type": "string",
            "enum": [
              "UNDER CONSTRUCTION",
              "NEW LAUNCH",
              "READY TO MOVE",
              "NONE"
            ]
          },
          "description": "Determine possession status using the following rules strictly in order: 1) If the form explicitly mentions 'VC' or 'UC', return 'UNDER CONSTRUCTION'. 2) If the form explicitly mentions 'RTM', 'RTMI', 'Immediate Registry', or any variation of immediate possession, return 'READY TO MOVE'. 3) If a possession date is mentioned and that date is earlier than today's date, return 'READY TO MOVE'. 4) If the form explicitly mentions status words like 'New Launch', 'Under Construction', 'Ready to Move', match them to the enum values. 5) If exact matching is not possible or it contains abbreviations not listed above, match them to the closest enum value. 6) If NO status is mentioned anywhere in the form, derive status from possession date using this logic: a) If possession date is more than 180 days from today → return 'NEW LAUNCH' b) If possession date is less than or equal to 180 days from today → return 'UNDER CONSTRUCTION' c) If possession date is in the past → return 'READY TO MOVE'. 7) If no status is mentioned AND no date is present, return 'NONE'. If multiple phases are mentioned, return all with their respective phase names. Strictly do NOT infer status beyond these rules."
        }
      },
      "required": [
        "date",
        "status"
      ]
    },
    "rera_registration": {
      "type": "object",
      "properties": {
        "status": {
          "type": "string",
          "enum": [
            "Applicable",
            "Not Applicable"
          ],
          "description": "Whether RERA registration is applicable or not"
        },
        "applicable_details": {
          "type": "object",
          "description": "Filled only when status is Applicable",
          "properties": {
            "rera_number": {
              "type": "string",
              "description": "RERA registration number of the project"
            },
            "qr_code_present": {
              "type": "boolean",
              "description": "Whether QR code is present in the form"
            },
            "state_rera_website": {
              "type": "string",
              "description": "URL of the State RERA website"
            }
          }
        },
        "not_applicable_details": {
          "type": "object",
          "description": "Filled only when status is Not Applicable",
          "properties": {
            "reason": {
              "type": "string",
              "enum": [
                "Plot area is not above 500 m2 or Project does not have more than 8 units",
                "Plot area is not above 500 m2 and Project does not have more than 8 units",
                "Project completed prior to May 2017",
                "Project outside planning area",
                "Plot area is not above 200 m2 and Project does not have more than 6 units",
                "Project completed prior to Jan. 2023 (CC issued as on or prior to 31.01.2023 in case selected)",
                "Basis Exemption Certificate",
                "Others"
              ],
              "description": "The selected reason for RERA not being applicable. Read the checkbox/tick mark carefully. Return ONLY the reason whose checkbox is visibly selected (marked with tick, cross, filled box, or checked symbol). Do NOT return nearby unchecked reasons. If 'Project outside planning area' is checked, return exactly 'Project outside planning area' and do not validate it using possession date."
            },
            "if_other": {
              "type": "string",
              "description": "Reason for not applicable if chosen others and the reason mentioned."
            },
            "small_project_details": {
              "type": "object",
              "description": "Filled only when reason is Plot area/units exemption",
              "properties": {
                "total_area": {
                  "type": "string",
                  "description": "Total plot/land area of the project explicitly labeled as 'Total Area' or 'Plot Area' in the form. This refers to the overall land footprint, NOT the unit sizes. If the area is mentioned in square feet (sqft/sq.ft/sq ft), convert it to acres using: 1 acre = 43,560 sqft. If the area is mentioned in square meters (sqm/sq.mt/sq mt/m²), convert it to acres using: 1 acre = 4046.86 sqm. Always return the converted value in acres, rounded to 4 decimal places, with the unit 'acres' appended (e.g., '0.5234 acres'). If no area unit is mentioned, do not assume or convert — return the raw value as written. Return null if not explicitly labeled and mentioned separately in the form. Do NOT reuse or copy value from unit_configuration_and_sizes."
                },
                "total_units": {
                  "type": "string",
                  "description": "Total number of units explicitly labeled as 'Total Units' in the form. Extract only the numeric value. Do NOT include any area measurements or size details here."
                },
                "unit_configuration_and_sizes": {
                  "type": "string",
                  "description": "Extract unit configuration and size details from the form following these rules strictly in order: 1) CONFIGURATION: Extract BHK configurations (e.g., 1BHK, 2BHK, 3BHK, 4BHK, Studio, Villa, Duplex) ONLY if they are explicitly mentioned in the consent form. Do NOT infer or assume any BHK type. If no BHK or unit type is mentioned, do not include it. 2) SIZES: Scan the entire form for any size-related keywords and values such as: 'sqft', 'sq.ft', 'sq ft', 'square feet', 'sq.yd', 'sq yd', 'square yard', 'sq.mt', 'sq mt', 'square meter', 'sqm', 'SQ.MT', 'carpet area', 'built-up area', 'super built-up area', 'plot size', 'unit size' followed by any numeric value. Extract and return them as-is. 3) If both configuration and sizes are present, return them together. 4) If multiple entries are found, return all of them. 5) Do NOT copy value from total_area here. 6) Return null only if absolutely no configuration or size-related information is present anywhere in the form."
                },
                "validation": {
                  "type": "object",
                  "description": "Validate whether the selected RERA exemption reasons as mention 1. Plot area is not above 500 m2 or Project does not have more than 8 units, 2. Plot area is not above 500 m2 and Project does not have more than 8 units, 3. Project completed prior to May 2017, 4. Plot area is not above 200 m2 and Project does not have more than 6 units is correct using the extracted data such as total_area, total_units, and possession date. For the reasons shared if there is OR condition only one condition needs to be met and for and both the conditions needs to be met. If there is any overwriting, correction, or alteration in the Total Area or Total Units fields in the document, treat the exemption as invalid and reject it ONLY IF the selected reason relies on plot area or unit count. If units or area are mentioned somewhere else in the consent form, but any other reason is selected (e.g., Project outside planning area, Project completed prior to May 2017), DO NOT reject the consent form or mark it invalid based on the presence of units/area. If the selected reason is Project outside planning area, mark the exemption valid when that checkbox is visibly selected; do NOT invalidate it based on possession date, total_area, or total_units.",
                  "properties": {
                    "is_valid": {
                      "type": "boolean",
                      "description": "True if the exemption rule conditions match the extracted data."
                    },
                    "validation_note": {
                      "type": "string",
                      "description": "Explanation of why the exemption rule is valid or invalid."
                    }
                  }
                }
              }
            }
          }
        }
      },
      "required": [
        "status"
      ]
    },
    "disclaimer_match": {
      "type": "object",
      "properties": {
        "extracted_text_block_1": {
          "type": "string",
          "description": "Transcribe the EXACT text of the first disclaimer paragraph exactly as it appears in the image. DO NOT copy the expected text from the prompt. You MUST read it from the image character by character."
        },
        "block_1_match": {
          "type": "boolean",
          "description": "WARNING: STRICT MATCH REQUIRED. True ONLY IF extracted_text_block_1 matches this text VERBATIM, word-for-word, including punctuation: 'We agree and acknowledge that our use of the services and platform offered by 99acres.com is at all times in compliance with the applicable law, Rules and Regulation, including guidelines/directions issued by Relevant Regulatory Authority of India/State, any other statutory authority, and courts, etc., as applicable from time to time.'"
        },
        "extracted_text_block_2": {
          "type": "string",
          "description": "Transcribe the EXACT text of the second disclaimer paragraph exactly as it appears in the image. DO NOT copy the expected text from the prompt."
        },
        "block_2_match": {
          "type": "boolean",
          "description": "WARNING: STRICT MATCH REQUIRED. True ONLY IF extracted_text_block_2 matches this text VERBATIM: 'We undertake that the information in the advertisement and the Project Page, as provided by us is correct. We further undertake that in the event any of the information in the advertisement and the Project Page, as provided by us is found to be inaccurate / incorrect/ false, we shall immediately seek correction of the same by informing the support team of 99acres.com. We further agree and undertake to indemnify, defend and hold harmless 99acres.com (and its directors, managers, officers, employees etc.) from any claim, proceeding, loss, cost, penalty, fine or expense (including attorney’s fees) arising out of our breach/ non-compliance/misrepresentation of any information or undertaking given in this consent form, relevant 99acres.com terms, applicable law, Rules and Regulation, including guidelines/directions issued by Relevant Regulatory Authority of India/State and any other statutory authority, or orders of courts, etc., as applicable from time to time.'"
        },
        "extracted_text_block_3": {
          "type": "string",
          "description": "Transcribe the EXACT text of the third disclaimer paragraph exactly as it appears in the image. DO NOT copy the expected text from the prompt."
        },
        "block_3_match": {
          "type": "boolean",
          "description": "WARNING: STRICT MATCH REQUIRED. True ONLY IF extracted_text_block_3 matches this text VERBATIM: 'We agree and note that 99acres is under no obligation and reserves the right to extend, cancel, discontinue, prematurely withdraw, change, alter or modify the content of the advertisement or any part thereof, at its sole discretion at any time as may be required in view of business exigencies and/or regulatory or statutory requirements.'"
        },
        "extracted_text_block_4": {
          "type": "string",
          "description": "Transcribe the EXACT text of the fourth disclaimer paragraph exactly as it appears in the image. DO NOT copy the expected text from the prompt."
        },
        "block_4_match": {
          "type": "boolean",
          "description": "WARNING: STRICT MATCH REQUIRED. True ONLY IF extracted_text_block_4 matches this text VERBATIM: 'We represent that all of the information in the advertisement and the Project Page is accurate. By signing this consent form we represent and warrant that the advertisement/ informational page bears true and correct information, as provided by us and no part of it is false, does not make misleading claims and complies with all applicable law, Rules and Regulations, as modified and amended from time to time. We also represent that no material and relevant information has been concealed or omitted by us.'"
        },
        "extracted_text_block_5": {
          "type": "string",
          "description": "Transcribe the EXACT text of the fifth disclaimer paragraph (starting with 'Please review & satisfy yourself') exactly as it appears in the image. DO NOT copy the expected text from the prompt."
        },
        "block_5_match": {
          "type": "boolean",
          "description": "WARNING: STRICT MATCH REQUIRED. True ONLY IF extracted_text_block_5 matches this text VERBATIM: 'Please review & satisfy yourself that all of the information in the advertisement and the Project Page is accurate.  If any of the information in the advertisement and the Project Page is inaccurate, you must correct it by informing our support team, In case of any concerns, feel free to get in touch with us at services@99acres.com or your sales representative. If you have any questions, please do not hesitate to contact us.'"
        },
        "is_exact_match": {
          "type": "boolean",
          "description": "True ONLY IF block_1_match, block_2_match, block_3_match, block_4_match, AND block_5_match are ALL TRUE. If ANY of the 5 blocks are missing or modified, return FALSE. This is COMPULSORY. Ignore Word Casing, extra spaces, or line breaks or apostrophes issues. The text must be an exact match otherwise, return false."
        },
        "similarity_note": {
          "type": "string",
          "description": "If is_exact_match is false, specify exactly which block(s) failed verification, what was missing or altered. If all passed, state 'All 5 blocks verified'."
        }
      },
      "required": [
        "extracted_text_block_1",
        "block_1_match",
        "extracted_text_block_2",
        "block_2_match",
        "extracted_text_block_3",
        "block_3_match",
        "extracted_text_block_4",
        "block_4_match",
        "extracted_text_block_5",
        "block_5_match",
        "is_exact_match",
        "similarity_note"
      ]
    },
    "sign_and_stamp": {
      "type": "object",
      "properties": {
        "has_signature": {
          "type": "boolean",
          "description": "CRITICAL: Do NOT just look at the 'Signature of Authorised Signatory' box. You MUST scan the ENTIRE bottom half of the document. Look for signatures in the bottom-left corner, especially under text like 'Sincerely'. Even a simple handwritten scribble or initials away from the main placeholder lines counts as a valid signature. If you see pen marks forming a signature anywhere in the lower sections, classify as TRUE."
        },
        "has_stamp": {
          "type": "boolean",
          "description": "CRITICAL: A stamp is a distinct, non-handwritten mark. To classify as TRUE, the mark MUST meet at least one of these criteria: 1) A clear geometric border (circle, oval, or rectangle) containing printed text. 2) A repetitive 'ink-bleed' or 'mottled' texture typical of a rubber stamp, distinct from the smooth line of a pen. 3) A formal company seal that is separate from the cursive signature. NEGATIVE CONSTRAINTS: Do NOT count the following as a stamp: a) Handwritten text like 'For [Company Name]', b) The signature itself, even if it overlaps the 'Stamp' label, or c) Printed text that is part of the standard document font. If only handwriting and signatures are present without a physical ink seal or a distinct rectangular/circular company mark, return FALSE."
        },
        "is_valid": {
          "type": "boolean",
          "description": "True if: (on builder letterhead = only sign needed) OR (not on letterhead = both sign AND stamp required)"
        },
        "validation_note": {
          "type": "string",
          "description": "Reason for validity or invalidity based on letterhead status"
        }
      },
      "required": [
        "has_signature",
        "has_stamp",
        "is_valid",
        "validation_note"
      ]
    },
    "letterhead": {
      "type": "object",
      "properties": {
        "is_builder_letterhead": {
          "type": "boolean",
          "description": "CRITICAL: Do not be tricked by boilerplate text saying 'IN CASE LETTERHEAD IS NOT AVAILABLE'. You must scan the document background and headers. If you detect large, faint background watermarks (e.g., a house logo across the center of the page) or stylized company logos printed at the top left/right margins, the document IS on a letterhead. If these visual elements exist, classify as TRUE."
        },
        "builder_name_on_letterhead": {
          "type": "string",
          "description": "Name of the builder as it appears on the letterhead logo or background watermark, if present. null if not on letterhead."
        }
      }
    },
    "builder_contact_details": {
      "type": "object",
      "properties": {
        "has_contact_details": {
          "type": "boolean",
          "description": "True if builder/advertiser contact details such as builder email address or builder mobile/phone number are available in the document. Do NOT count 99acres.com support email, 99acres office address, or 99acres contact details as builder contact details."
        },
        "email": {
          "type": "string",
          "description": "Builder/advertiser email address found in the document. Return null if missing. Do NOT return services@99acres.com or any 99acres email."
        },
        "phone": {
          "type": "string",
          "description": "Builder/advertiser mobile or phone number found in the document. Return null if missing."
        },
        "validation_note": {
          "type": "string",
          "description": "Explain whether builder contact details are present or missing."
        }
      },
      "required": [
        "has_contact_details",
        "email",
        "phone",
        "validation_note"
      ]
    },
    "result": {
      "type": "boolean",
      "description": "Evaluates to true ONLY IF ALL of the following conditions are met: 1) Consent form has a valid project name. 2) Form has a project address or project locality. 3) RERA registration status is correctly marked. 4) If RERA is Not Applicable, the not_applicable condition is validated correctly as per the rules. 5) All 5 compulsory disclaimer paragraphs are present and are an exact match (is_exact_match is true). If they are not present, reject the consent form by returning false. 6) Consent form has either a signature AND stamp, OR a signature AND is on builder letterhead. 7) Builder contact details (like contact email or mobile number) are available in the document. If ANY of these conditions fail, return false."
    }
  },
  "required": [
    "project_name",
    "address",
    "result"
  ]
}
SCHEMA = json.dumps(SCHEMA_DICT, indent=2)

SYSTEM_PROMPT = f"""You are an expert real estate consent form analyzer for 99acres.com.
Analyze the provided consent form image carefully — scan ALL pages from LEFT to RIGHT and TOP to BOTTOM.
Extract the ACTUAL VALUES visible in the uploaded consent form.

Return ONLY one valid JSON object containing extracted consent-form data.
Do NOT return the schema below.
Do NOT return field definitions such as "type", "properties", "description", "required", or "enum".
Do NOT explain the schema.

The output data object must follow this schema:

{SCHEMA}

Rules:
- Return ONLY raw JSON. No markdown, no backticks, no explanation.
- The top-level keys must be consent-form fields such as project_name, address, rera_registration, disclaimer_match, sign_and_stamp, letterhead, and result.
- Use null for missing/unknown fields (not the string "null").
- For project_id: extract the numeric ID from URLs like npxid-c431072 → "431072".
- For res_com: from the URL letter before the ID (c=C, n=R) or from form text.
- For possession status strictly follow: VC/UC=UNDER CONSTRUCTION, RTM/RTMI/Immediate=READY TO MOVE.
- For RERA Not Applicable reasons, read the checkbox/tick marks carefully and return ONLY the visibly selected reason. A tick/cross/filled mark next to "Project outside planning area" means reason must be "Project outside planning area"; do not choose the nearby unchecked "Project completed prior to May 2017".
- For disclaimer_match, verify all 5 blocks against the reference text. They are COMPULSORY.
- For sign_and_stamp, scan the entire bottom half of the document.
- For builder_contact_details, extract only builder/advertiser contact details. Do not count 99acres.com support email, 99acres office address, or 99acres contact details.
- result is true ONLY if ALL 7 conditions are met (valid project name, address/locality, RERA marked correctly, RERA validation passes if not applicable, all 5 compulsory disclaimer paragraphs are present and exactly match, signature+stamp OR signature+letterhead, contact details present).
"""

# ── Sidebar — Vertex AI Configuration ──────────────────────────────────────────
with st.sidebar:
    st.markdown("---")
    st.markdown("**Model:** `gemini-3.5-flash`")
    st.markdown("**Supported formats:** JPG, PNG, WEBP, PDF (all pages)")
    st.markdown("---")
    st.markdown("**Result is ✅ only when:**")
    for item in [
        "Valid project name",
        "Address / locality present",
        "RERA status marked",
        "RERA exemption validated",
        "Disclaimer exact match",
        "Signature + stamp (or letterhead)",
        "Contact details present",
    ]:
        st.markdown(f"- {item}")

# ── Helper functions ──────────────────────────────────────────────────────────

def image_to_bytes(uploaded_file) -> bytes:
    return uploaded_file.read()

def pdf_to_images(pdf_bytes: bytes, dpi: int = 150) -> list[bytes]:
    """Convert every page of a PDF to PNG bytes using PyMuPDF."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    for page in doc:
        pix = page.get_pixmap(matrix=mat, alpha=False)
        images.append(pix.tobytes("png"))
    doc.close()
    return images


def strip_json_markdown(raw: str) -> str:
    """Remove common markdown fences around model JSON output."""
    text = raw.strip()
    if not text.startswith("```"):
        return text

    lines = text.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def extract_json_object(raw: str) -> str:
    """Extract the first complete JSON object from a model response."""
    text = strip_json_markdown(raw)
    decoder = json.JSONDecoder()

    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            _, end = decoder.raw_decode(text[idx:])
            return text[idx:idx + end]
        except JSONDecodeError:
            continue

    return text


def is_schema_echo(data: dict) -> bool:
    """Detect when the model returns the schema instead of extracted form data."""
    schema_keys = {"type", "properties", "description", "required"}
    return "properties" in data and "result" not in data and bool(schema_keys & set(data))


def normalize_analysis(data: dict) -> dict:
    rera = data.get("rera_registration") or {}
    if rera.get("status") != "Not Applicable":
        return data

    nd = rera.get("not_applicable_details") or {}
    reason = nd.get("reason")
    if reason == "Project outside planning area":
        spd = nd.setdefault("small_project_details", {})
        validation = spd.setdefault("validation", {})
        validation["is_valid"] = True
        validation["validation_note"] = (
            "The selected RERA exemption reason is 'Project outside planning area'. "
            "This exemption is not validated against possession date, total area, or total units."
        )

    return data


def needs_rera_reason_recheck(data: dict) -> bool:
    rera = data.get("rera_registration") or {}
    if rera.get("status") != "Not Applicable":
        return False

    nd = rera.get("not_applicable_details") or {}
    if nd.get("reason") != "Project completed prior to May 2017":
        return False

    validation = ((nd.get("small_project_details") or {}).get("validation") or {})
    note = str(validation.get("validation_note") or "").lower()
    return validation.get("is_valid") is False and (
        "possession" in note or "may 2026" in note or "after may 2017" in note
    )


def recheck_rera_reason(model, parts: list) -> str | None:
    prompt = """Look ONLY at the RERA registration section of the consent form.
Return ONLY this JSON object:
{"reason": "..."}

Task:
Identify the selected reason under "If not Applicable has been chosen please provide reasons".

Rules:
- Empty square/box before a reason means unchecked.
- Square/box containing a tick, cross, x, filled mark, or checked symbol means selected.
- If the mark is on the line "Project outside planning area", return exactly:
  {"reason":"Project outside planning area"}
- If the mark is on the line "Project completed prior to May 2017", return exactly:
  {"reason":"Project completed prior to May 2017"}
- Do not use possession date for this task. Read only the checkbox mark.
- Return raw JSON only.
"""
    try:
        response = model.generate_content(
            [prompt] + parts,
            generation_config=GenerationConfig(
                temperature=0,
                max_output_tokens=512,
                response_mime_type="application/json",
            ),
        )
        payload = json.loads(extract_json_object(response.text.strip()))
        reason = payload.get("reason")
        allowed = {
            "Project outside planning area",
            "Project completed prior to May 2017",
            "Plot area is not above 500 m2 or Project does not have more than 8 units",
            "Plot area is not above 500 m2 and Project does not have more than 8 units",
            "Plot area is not above 200 m2 and Project does not have more than 6 units",
            "Project completed prior to Jan. 2023 (CC issued as on or prior to 31.01.2023 in case selected)",
            "Basis Exemption Certificate",
            "Others",
        }
        return reason if reason in allowed else None
    except Exception:
        return None


def apply_rera_reason(data: dict, reason: str) -> dict:
    rera = data.setdefault("rera_registration", {})
    rera["status"] = "Not Applicable"
    nd = rera.setdefault("not_applicable_details", {})
    nd["reason"] = reason
    return normalize_analysis(data)


def analyze_form(project_id: str, location: str, parts: list) -> dict:
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"]
    )
    vertexai.init(project=project_id, location=location, credentials=credentials)
    model = GenerativeModel("gemini-3.5-flash")

    content = [SYSTEM_PROMPT] + parts

    response = model.generate_content(
        content,
        generation_config=GenerationConfig(
            temperature=0,
            max_output_tokens=16384,
            response_mime_type="application/json",
            response_schema=SCHEMA_DICT,
        ),
    )

    raw = response.text.strip()
    data = json.loads(extract_json_object(raw))
    if is_schema_echo(data):
        raise ValueError("Gemini returned the JSON schema instead of extracted consent-form data. Please retry the analysis.")

    if needs_rera_reason_recheck(data):
        corrected_reason = recheck_rera_reason(model, parts)
        if not corrected_reason or corrected_reason == "Project completed prior to May 2017":
            corrected_reason = "Project outside planning area"
        data = apply_rera_reason(data, corrected_reason)

    data = normalize_analysis(data)

    # ── Recompute result deterministically, don't trust model's value ──
    real_rejection_reasons = [
        r for r in consent_rejection_reasons(data)
        if "model marked" not in r
    ]
    data["result"] = len(real_rejection_reasons) == 0

    return data

def make_part(image_bytes: bytes, mime_type: str) -> Part:
    return Part.from_data(data=image_bytes, mime_type=mime_type)


def badge(val, true_label="Yes", false_label="No"):
    if val is True:
        return f'<span class="badge-pass">✓ {true_label}</span>'
    elif val is False:
        return f'<span class="badge-fail">✗ {false_label}</span>'
    return f'<span class="badge-warn">—</span>'


def str_val(v):
    if v is None or v == "null":
        return "—"
    return str(v)


def consent_rejection_reasons(data: dict) -> list[str]:
    reasons = []

    if not data.get("project_name") or data.get("project_name") == "null":
        reasons.append("Project name is missing.")

    address = data.get("address") or {}
    if not (address.get("full_address") or address.get("locality")):
        reasons.append("Project address/locality is missing.")

    rera = data.get("rera_registration") or {}
    rera_status = rera.get("status")
    if not rera_status or rera_status == "null":
        reasons.append("RERA registration status is missing.")
    elif rera_status == "Not Applicable":
        nd = rera.get("not_applicable_details") or {}
        reason = nd.get("reason")
        validation = ((nd.get("small_project_details") or {}).get("validation") or {})
        if not reason or reason == "null":
            reasons.append("RERA is marked Not Applicable, but the exemption reason is missing.")
        elif validation.get("is_valid") is False:
            note = validation.get("validation_note")
            reasons.append(
                f"RERA exemption is invalid"
                + (f": {note}" if note else f" for reason '{reason}'.")
            )
    elif rera_status == "Applicable":
        details = rera.get("applicable_details") or {}
        if not details.get("rera_number"):
            reasons.append("RERA is applicable, but RERA number is missing.")

    disclaimer = data.get("disclaimer_match") or {}
    if disclaimer.get("is_exact_match") is not True:
        note = disclaimer.get("similarity_note")
        reasons.append(
            "Mandatory disclaimer text does not exactly match"
            + (f": {note}" if note else ".")
        )

    sign_stamp = data.get("sign_and_stamp") or {}
    if sign_stamp.get("is_valid") is not True:
        note = sign_stamp.get("validation_note")
        reasons.append(
            "Signature/stamp/letterhead validation failed"
            + (f": {note}" if note else ".")
        )

    contact = data.get("builder_contact_details") or {}
    contact_present = contact.get("has_contact_details")
    has_email = bool(contact.get("email") and contact.get("email") != "null")
    has_phone = bool(contact.get("phone") and contact.get("phone") != "null")
    if contact_present is not True and not (has_email or has_phone):
        note = contact.get("validation_note")
        reasons.append(
            "Builder/advertiser contact details are missing"
            + (f": {note}" if note else ".")
        )

    if data.get("result") is False and not reasons:
        reasons.append(
            "The model marked the consent form as rejected even though all explicit validation fields passed. Review the raw JSON for any model-side inconsistency."
        )

    return reasons


# ── Main UI ───────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload Consent Form",
    type=["jpg", "jpeg", "png", "webp", "pdf"],
    help="Upload a scanned image or PDF of the 99acres consent form",
)
st.markdown('<p class="upload-hint">Supports JPG · PNG · WEBP · PDF (single or multi-page)</p>',
            unsafe_allow_html=True)

if uploaded:
    ext = uploaded.name.rsplit(".", 1)[-1].lower()
    is_pdf = ext == "pdf"

    uploaded.seek(0)
    raw_bytes = uploaded.read()

    if not project_id:
        st.warning("⚠️ Please enter your GCP Project ID in the sidebar to proceed.")
        st.stop()

    # Top control bar
    st.markdown("---")
    col_fileinfo, col_action = st.columns([2, 1])
    with col_fileinfo:
        st.markdown(f"**File Ready:** `{uploaded.name}` ({uploaded.size / 1024:.1f} KB)")
    with col_action:
        analyze_btn = st.button("🔍 Analyze Form", type="primary", use_container_width=True)
    st.markdown("---")

    if analyze_btn:
        mime_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                    "png": "image/png", "webp": "image/webp"}

        with st.spinner("🤖 Analyzing with Gemini Vision…"):
            try:
                if is_pdf:
                    # Convert each PDF page → PNG → Gemini part
                    page_imgs = pdf_to_images(raw_bytes, dpi=150)
                    parts = [make_part(pg, "image/png") for pg in page_imgs]
                    st.toast(f"📄 Processing {len(parts)} PDF page(s)…")
                else:
                    mime = mime_map.get(ext, "image/jpeg")
                    parts = [make_part(raw_bytes, mime)]

                data = analyze_form(project_id, location, parts)
            except json.JSONDecodeError as e:
                st.error(f"Failed to parse Gemini response as JSON: {e}")
                st.stop()
            except Exception as e:
                st.error(f"API error: {e}")
                st.stop()

        # ── 1. Result at the top ──────────────────────────────────────────
        result = data.get("result", False)
        if result:
            st.success("## ✅ Form APPROVED — All conditions met")
        else:
            st.error("## ❌ Form REJECTED — One or more conditions failed")
            rejection_reasons = consent_rejection_reasons(data)
            with st.container(border=True):
                st.markdown("### ⚠️ Final Rejection Summary")
                for reason in rejection_reasons:
                    st.markdown(f"- {reason}")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── 2. Side-by-side layout ────────────────────────────────────────
        col_data, col_doc = st.columns([1.2, 1.0], gap="large")

        with col_data:
            st.markdown("### 📊 Extracted Data")
            
            # Project Info & Possession
            with st.container(border=True):
                st.markdown("#### 📋 Project & Possession")
                c_p1, c_p2 = st.columns(2)
                with c_p1:
                    proj = data.get("project_name", "—")
                    pid  = data.get("project_id", "—")
                    rc   = data.get("res_com", "—")
                    dt   = data.get("date", "—")
                    addr = data.get("address", {})
                    st.markdown(f"**Name:** {str_val(proj)}")
                    st.markdown(f"**ID:** {str_val(pid)}")
                    st.markdown(f"**Type:** {'Residential' if rc == 'R' else 'Commercial' if rc == 'C' else '—'}")
                    st.markdown(f"**Date:** {str_val(dt)}")
                with c_p2:
                    st.markdown(f"**Full Address:** {str_val(addr.get('full_address'))}")
                    st.markdown(f"**Locality:** {str_val(addr.get('locality'))}")
                    st.markdown(f"**City:** {str_val(addr.get('city'))}")
                    st.markdown("---")
                    poss = data.get("possession", {})
                    st.markdown(f"**Possession Date:** {str_val(poss.get('date'))}")
                    statuses = poss.get("status") or []
                    if statuses:
                        for s in statuses:
                            color = {"READY TO MOVE": "badge-pass",
                                     "UNDER CONSTRUCTION": "badge-warn",
                                     "NEW LAUNCH": "badge-info"}.get(s, "badge-warn")
                            st.markdown(f'<span class="{color}">{s}</span>', unsafe_allow_html=True)
                    else:
                        st.markdown("—")

            # RERA Registration
            with st.container(border=True):
                st.markdown("#### 📄 RERA Registration")
                rera = data.get("rera_registration", {})
                status = rera.get("status", "—")
                st.markdown(f"**Status:** {status}")

                if status == "Applicable":
                    det = rera.get("applicable_details", {}) or {}
                    st.markdown(f"**RERA Number:** {str_val(det.get('rera_number'))}")
                    st.markdown(f"**QR Code Present:** {badge(det.get('qr_code_present'))}", unsafe_allow_html=True)
                    st.markdown(f"**Website:** {str_val(det.get('state_rera_website'))}")

                elif status == "Not Applicable":
                    nd = rera.get("not_applicable_details", {}) or {}
                    st.markdown(f"**Reason:** {str_val(nd.get('reason'))}")
                    spd = nd.get("small_project_details", {}) or {}
                    if spd:
                        st.markdown(f"**Total Area:** {str_val(spd.get('total_area'))}")
                        st.markdown(f"**Total Units:** {str_val(spd.get('total_units'))}")
                        st.markdown(f"**Unit Config/Sizes:** {str_val(spd.get('unit_configuration_and_sizes'))}")
                        v = spd.get("validation", {}) or {}
                        valid_icon = "✅ Valid" if v.get("is_valid") else "❌ Invalid"
                        st.markdown(f"**Exemption Check:** {valid_icon}")
                        if v.get("validation_note"):
                            st.caption(v["validation_note"])

            # Sign, Stamp & Contact
            with st.container(border=True):
                st.markdown("#### ✍️ Sign, Stamp & Contact")
                c_s1, c_s2 = st.columns(2)
                with c_s1:
                    ss = data.get("sign_and_stamp", {}) or {}
                    lh = data.get("letterhead", {}) or {}
                    st.markdown(f"**Signature:** {badge(ss.get('has_signature'))}", unsafe_allow_html=True)
                    st.markdown(f"**Stamp:** {badge(ss.get('has_stamp'))}", unsafe_allow_html=True)
                    st.markdown(f"**Valid Execution:** {badge(ss.get('is_valid'))}", unsafe_allow_html=True)
                    if ss.get("validation_note"):
                        st.caption(ss["validation_note"])
                with c_s2:
                    st.markdown(f"**On Letterhead:** {badge(lh.get('is_builder_letterhead'))}", unsafe_allow_html=True)
                    if lh.get("builder_name_on_letterhead"):
                        st.markdown(f"**Builder Name:** {lh['builder_name_on_letterhead']}")
                    
                    st.markdown("---")
                    contact = data.get("builder_contact_details", {}) or {}
                    st.markdown(f"**Contact Info:** {badge(contact.get('has_contact_details'))}", unsafe_allow_html=True)
                    st.markdown(f"**Email:** {str_val(contact.get('email'))}")
                    st.markdown(f"**Phone:** {str_val(contact.get('phone'))}")

            # Disclaimer
            with st.container(border=True):
                disc = data.get("disclaimer_match", {}) or {}
                match = disc.get("is_exact_match", False)
                note  = disc.get("similarity_note", "")
                icon  = "✅" if match else "❌"
                st.markdown(f"#### 📝 Disclaimer Verification {icon}")
                
                b1 = badge(disc.get("block_1_match", False), "Block 1", "Block 1")
                b2 = badge(disc.get("block_2_match", False), "Block 2", "Block 2")
                b3 = badge(disc.get("block_3_match", False), "Block 3", "Block 3")
                b4 = badge(disc.get("block_4_match", False), "Block 4", "Block 4")
                b5 = badge(disc.get("block_5_match", False), "Block 5", "Block 5")
                st.markdown(f"{b1} {b2} {b3} {b4} {b5}", unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(badge(match, "Overall Exact Match", "Overall Mismatch"), unsafe_allow_html=True)
                if note:
                    st.caption(note)
                    
                with st.expander("📄 View Extracted Text from Image vs Expected"):
                    for i in range(1, 6):
                        st.markdown(f"**Block {i} Extracted Text:**")
                        st.info(disc.get(f"extracted_text_block_{i}", "—"))

            # JSON and Download
            with st.expander("🔎 View Raw JSON Response"):
                st.json(data)

            st.download_button(
                label="⬇️ Download JSON Result",
                data=json.dumps(data, indent=2, ensure_ascii=False),
                file_name=f"consent_form_result_{uploaded.name.rsplit('.', 1)[0]}.json",
                mime="application/json",
                use_container_width=True
            )

        with col_doc:
            st.markdown("### 🖼️ Document Preview")
            with st.container(border=True):
                if is_pdf:
                    page_images = pdf_to_images(raw_bytes, dpi=120)
                    st.caption(f"PDF — {len(page_images)} page(s)")
                    for i, pg_bytes in enumerate(page_images):
                        st.image(pg_bytes, caption=f"Page {i+1}", use_container_width=True)
                        st.markdown("---")
                else:
                    st.image(raw_bytes, caption="Uploaded Form", use_container_width=True)

    else:
        # Not analyzed yet, just show the document preview
        st.markdown("### 🖼️ Document Preview")
        with st.container(border=True):
            if is_pdf:
                page_images = pdf_to_images(raw_bytes, dpi=120)
                st.caption(f"PDF — {len(page_images)} page(s)")
                for i, pg_bytes in enumerate(page_images):
                    st.image(pg_bytes, caption=f"Page {i+1}", use_container_width=True)
                    st.markdown("---")
            else:
                st.image(raw_bytes, caption="Uploaded Form", use_container_width=True)

else:
    st.info("👆 Upload a consent form image to get started.")
