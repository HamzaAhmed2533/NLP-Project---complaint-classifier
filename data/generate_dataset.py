"""
generate_dataset.py
-------------------
Utility script to create a sample customer complaints CSV dataset
for demonstration and testing purposes.

Run this once if you do not have a real dataset:
    python data/generate_dataset.py
"""

import csv
import random
import os

# ---------------------------------------------------------------------------
# Seed for reproducibility
# ---------------------------------------------------------------------------
random.seed(42)

# ---------------------------------------------------------------------------
# Complaint templates per category
# ---------------------------------------------------------------------------
COMPLAINTS = {
    "billing_issue": [
        "I was charged twice for the same order and need a refund immediately.",
        "My bill shows extra charges that were never explained to me.",
        "The invoice I received does not match the amount I was quoted.",
        "I have been overcharged on my monthly statement again.",
        "There is an unauthorized transaction on my account.",
        "I requested a refund three weeks ago and still have not received it.",
        "The discount coupon was not applied to my bill.",
        "My payment was processed but the order shows as unpaid.",
        "I received a bill for a service I never signed up for.",
        "The late fee on my account was applied unfairly.",
        "I cannot understand the breakdown of charges on my invoice.",
        "My credit card was debited but no confirmation email was sent.",
        "I was billed at the old price after a plan change.",
        "The subscription fee was renewed without my consent.",
        "I need an itemized bill for my last three purchases.",
    ],
    "delivery_problem": [
        "My package has not arrived even though tracking shows it was delivered.",
        "The item I ordered was delivered to the wrong address.",
        "My delivery is three days late with no update from the courier.",
        "The package arrived completely damaged due to poor packaging.",
        "I never received the shipping confirmation for my order.",
        "The estimated delivery date keeps being pushed back.",
        "Half of my order was missing when the parcel arrived.",
        "The courier attempted delivery without ringing the doorbell.",
        "My order was marked returned without any delivery attempt.",
        "Items in my shipment were broken because of inadequate padding.",
        "I have been waiting over two weeks for a standard delivery.",
        "The tracking number provided does not work on the courier website.",
        "My express delivery was shipped via standard delivery instead.",
        "The parcel was left in an unsecured location and is now missing.",
        "I paid for same-day delivery but received it three days later.",
    ],
    "product_defect": [
        "The product stopped working after just two days of use.",
        "The item I received has a visible crack right out of the box.",
        "The color of the product is completely different from what I ordered.",
        "The device does not power on even after following setup instructions.",
        "The zipper on the bag broke within the first week.",
        "There are scratches and marks on the product despite brand-new packaging.",
        "The size I received does not match the size I ordered online.",
        "The product has a strange odor that does not go away.",
        "The buttons on the device are stuck and unresponsive.",
        "The fabric shrank significantly after the first wash.",
        "The product manual describes features that are not present on my unit.",
        "The battery drains completely within an hour of full charge.",
        "The screen has dead pixels and is unusable.",
        "Several components were missing from the product box.",
        "The product did not match the description or photos on the website.",
    ],
    "service_complaint": [
        "The customer support representative was rude and unhelpful.",
        "I waited on hold for over an hour and my issue was never resolved.",
        "No one has responded to my email complaint submitted last week.",
        "The technician who visited my home did not fix the problem.",
        "I was promised a callback that never happened.",
        "The support team closed my ticket without resolving the issue.",
        "I was transferred between five departments and still got no help.",
        "The live chat disconnected multiple times during my complaint.",
        "The agent gave me incorrect information about my warranty.",
        "My account was locked and customer service cannot explain why.",
        "I have been trying to cancel my subscription for two months.",
        "The service was interrupted without any prior notice.",
        "Staff at the store were dismissive and unwilling to assist.",
        "I received no acknowledgment after submitting my complaint form.",
        "The refund process was explained incorrectly by the support team.",
    ],
}

# ---------------------------------------------------------------------------
# Build records
# ---------------------------------------------------------------------------
records = []
for label, texts in COMPLAINTS.items():
    for text in texts:
        records.append({"text": text, "label": label})

# Shuffle records
random.shuffle(records)

# ---------------------------------------------------------------------------
# Write CSV
# ---------------------------------------------------------------------------
output_path = os.path.join(os.path.dirname(__file__), "complaints.csv")

with open(output_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["text", "label"])
    writer.writeheader()
    writer.writerows(records)

print(f"Dataset saved to: {output_path}")
print(f"Total records: {len(records)}")
print("Label distribution:")
for label, texts in COMPLAINTS.items():
    print(f"  {label}: {len(texts)} samples")
