"""Invented names for customers, suppliers and items.

Built from made-up place-like stems and ordinary trade words, so they read like
British business names without being copied from any real register. Any
resemblance to a real company is coincidence.
"""

from __future__ import annotations

STEMS = [
    "Ashwell", "Brackmoor", "Calderby", "Dunstall", "Elmbridge", "Farrowdale", "Glenholme", "Harcliffe",
    "Ivybank", "Kestermere", "Larkfield", "Marlowe", "Netherby", "Oakhurst", "Pennick", "Quarrington",
    "Ravensholt", "Saltmere", "Thornbury", "Underhill", "Valemont", "Westerdale", "Yarrowby", "Ambleton",
    "Birchgrove", "Coldharbour", "Daleford", "Eastwick", "Fenwold", "Greystone", "Hollinwood", "Kingsmere",
    "Longridge", "Millbrook", "Northgate", "Orchardleigh", "Pinewood", "Redcastle", "Stanwick", "Tidebrook",
    "Upwell", "Wexcombe", "Brookhaven", "Carrowmore", "Deepdale", "Easterbrook", "Foxley", "Gorsefield",
    "Highcroft", "Ironbridge", "Kelsham", "Loxwood", "Moorside", "Norbury", "Overton", "Pembrook",
]

TRADES = {
    "Construction": ["Construction", "Builders", "Civils", "Groundworks", "Contracting", "Scaffolding", "Roofing"],
    "Manufacturing": ["Engineering", "Fabrication", "Precision", "Plastics", "Components", "Machining", "Castings"],
    "Utilities": ["Water Services", "Power Networks", "Energy", "Utilities", "Gas Services"],
    "Public Sector": ["Borough Council", "District Council", "NHS Estates", "Academy Trust", "Fire Service"],
    "Distributors": ["Supplies", "Distribution", "Trade Centre", "Industrial Supplies", "Wholesale"],
    "Facilities Management": ["Facilities", "Property Services", "Maintenance", "Building Services", "FM"],
}
SUPPLIER_TRADES = ["Manufacturing", "Wholesale", "Logistics", "Fixings", "Bearings", "Electrical", "Safety", "Tools"]
SUFFIXES = ["Ltd", "Ltd", "Ltd", "Group", "& Sons", "Holdings"]

ITEMS = {
    "Fasteners": (["Hex Bolt", "Socket Cap Screw", "Nyloc Nut", "Flat Washer", "Coach Screw", "Threaded Rod"],
                  ["M6", "M8", "M10", "M12", "M16", "M20"]),
    "Bearings": (["Deep Groove Bearing", "Taper Roller Bearing", "Pillow Block", "Flange Bearing", "Needle Bearing"],
                 ["6204", "6205", "6206", "30205", "UCP205", "UCF206"]),
    "Power Transmission": (["V-Belt", "Timing Belt", "Roller Chain", "Sprocket", "Taper Lock Bush", "Shaft Coupling"],
                           ["SPZ", "SPA", "08B", "10B", "1210", "1610"]),
    "Electrical": (["Contactor", "Cable Gland", "MCB", "Junction Box", "LED Floodlight", "Cable Tie Pack"],
                   ["16A", "20A", "32A", "IP66", "50W", "300mm"]),
    "PPE": (["Safety Boot", "Hi-Vis Vest", "Nitrile Gloves", "Safety Glasses", "Ear Defenders", "Hard Hat"],
            ["Size 8", "Size 10", "Large", "XL", "Clear", "White"]),
    "Hand Tools": (["Torque Wrench", "Socket Set", "Hex Key Set", "Pliers", "Screwdriver Set", "Spanner Set"],
                   ["1/2in", "3/8in", "Metric", "Imperial", "VDE", "12pc"]),
    "Lubricants": (["Multipurpose Grease", "Chain Lube", "Cutting Oil", "Penetrating Spray", "Gear Oil"],
                   ["400g", "500ml", "1L", "5L", "EP2"]),
    "Pneumatics": (["Push-in Fitting", "Air Hose", "Solenoid Valve", "Air Cylinder", "Regulator", "Silencer"],
                   ["6mm", "8mm", "10mm", "1/4in", "3/8in", "32x50"]),
}


def company_name(rng, group: str, used: set[str]) -> str:
    for _ in range(200):
        name = f"{rng.choice(STEMS)} {rng.choice(TRADES[group])}"
        if not group == "Public Sector":
            name += f" {rng.choice(SUFFIXES)}"
        if name not in used:
            used.add(name)
            return name
    raise RuntimeError("ran out of unique customer names; add stems")


def supplier_name(rng, used: set[str]) -> str:
    for _ in range(200):
        name = f"{rng.choice(STEMS)} {rng.choice(SUPPLIER_TRADES)} {rng.choice(SUFFIXES)}"
        if name not in used:
            used.add(name)
            return name
    raise RuntimeError("ran out of unique supplier names; add stems")


def item_name(rng, group: str, i: int) -> str:
    nouns, sizes = ITEMS[group]
    return f"{nouns[i % len(nouns)]} {sizes[(i // len(nouns)) % len(sizes)]}" + (f" Pk{i // 36 + 1}" if i >= 36 else "")


# ------------------------------------------------------------------ references typed by people
# Built from how finance staff actually write them (examples from Joseph, 12 Nov 2026):
# "Credit raise for inv 73638", "partial credit for inv 683772",
# "credit for invoices 73827, 783728, 72728".

def typo(rng, number: int) -> str:
    """A mistyped document number: two digits swapped, one dropped, one doubled or one wrong."""
    s = str(number)
    kind = rng.choice(["swap", "drop", "double", "wrong"])
    i = rng.randrange(len(s) - 1)
    if kind == "swap" and s[i] != s[i + 1]:
        return s[:i] + s[i + 1] + s[i] + s[i + 2:]
    if kind == "drop":
        return s[:i] + s[i + 1:]
    if kind == "double":
        return s[:i] + s[i] + s[i:]
    wrong = str((int(s[i]) + rng.choice([1, 9])) % 10)
    return s[:i] + wrong + s[i + 1:]


def _join(rng, refs: list[str]) -> str:
    if len(refs) == 1:
        return refs[0]
    return rng.choice([", ", " / ", " & ", " "]).join(refs)


def credit_comment(rng, quality: str, invoice_numbers: list[int], customer_refs: list[str], partial: bool) -> str:
    """What someone typed in ORIN.Comments when raising a credit not based on the invoice."""
    many = len(invoice_numbers) > 1
    if quality == "none":
        return rng.choice(["Credit as agreed", "Goodwill credit", "Pricing adjustment", "Credit per email",
                           "Returns credit", "Credit note", ""])
    if quality == "po_only":
        return rng.choice(["Credit re PO {r}", "Credit against your order {r}", "Cr - order {r}",
                           "Credit re your orders {r}"]).format(r=_join(rng, customer_refs))
    nums = [str(n) for n in invoice_numbers]
    if quality == "typo":
        k = rng.randrange(len(nums))
        nums[k] = typo(rng, invoice_numbers[k])
    refs = _join(rng, nums)
    if many:
        word = rng.choice(["invoices", "invs", "inv", "inv's"])
        return rng.choice(["Credit for {w} {r}", "credit for {w} {r}", "Rebate credit {w} {r}",
                           "Credit raised against {w} {r}"]).format(w=word, r=refs)
    if partial:
        return rng.choice(["Partial credit for inv {r}", "partial credit for inv {r}", "Part credit inv {r} short del",
                           "cr re inv {r} short delivery", "Credit raise for inv {r} - price diff"]).format(r=refs)
    return rng.choice(["Credit raise for inv {r}", "Credit for inv {r}", "Credit for invoice {r}", "Cr inv {r}",
                       "Returned goods inv {r}", "Credit raised for invoive {r}"]).format(r=refs)


def remittance_comment(rng, quality: str, invoice_numbers: list[int], customer_refs: list[str], customer: str) -> str:
    """What came with a payment received on account: the remittance reference, if any."""
    short = customer.split()[0].upper()
    if quality == "none":
        return rng.choice([f"BACS {short}", f"{short} PAYMENT", "Faster payment", "BGC", ""])
    if quality == "po_only":
        return rng.choice(["Remit PO {r}", "Payment orders {r}", "{s} PO {r}"]).format(
            r=_join(rng, customer_refs), s=short)
    nums = [str(n) for n in invoice_numbers]
    if quality == "typo":
        k = rng.randrange(len(nums))
        nums[k] = typo(rng, invoice_numbers[k])
    return rng.choice(["Remittance inv {r}", "Payment for invoices {r}", "{s} INV {r}", "Inv {r} paid"]).format(
        r=_join(rng, nums), s=short)
