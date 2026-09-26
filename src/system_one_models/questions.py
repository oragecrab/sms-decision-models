"""Scam-triage question definitions and closed-choice categories."""

CATEGORIES = {
    "legitimate": "Routine account notices, appointment reminders, personal messages, or normal service bills.",
    "marketing": "Commercial sales, discounts, product promotions, or ordinary advertising.",
    "suspected_scam": "Likely phishing or fraud: suspicious fees, requests for secret codes or identity/financial details, or unsolicited remote access.",
}

INDUSTRIES = {
    "banking_finance": "Banks, financial accounts, investing, insurance, or personal finance.",
    "retail_ecommerce": "Shopping, online orders, marketplaces, or consumer products.",
    "shipping_logistics": "Parcel delivery, postal services, freight, or courier tracking.",
    "government_public_services": "Government agencies, taxes, benefits, water, electricity, utilities, or public services.",
    "technology_telecom": "Software, online accounts, devices, internet, or phone services.",
    "healthcare": "Healthcare providers, pharmacies, medical services, or health insurance.",
    "employment": "Jobs, recruiting, payroll, or workplace services.",
    "travel_hospitality": "Travel, accommodation, transport, or hospitality.",
    "other": "A recognizable industry that does not fit the listed categories.",
    "unclear": "There is not enough information to identify an industry.",
}

REQUESTED_ACTIONS = {
    "no_action": "No concrete action; informational or a notification only.",
    "reply_or_contact": "Reply, call, contact someone, unsubscribe, or continue elsewhere.",
    "click_or_scan": "Open a link, visit a website, or scan a QR code.",
    "sign_in_or_verify": "Sign in, reset a password, or verify an account or identity.",
    "share_sensitive_info_or_documents": "Provide passwords, security codes, personal or financial details, or documents.",
    "pay_or_transfer": "Pay a fee, bill, fine, or send money, gift cards, or cryptocurrency.",
    "purchase_or_claim": "Buy, subscribe, redeem an offer, or claim a prize, refund, or benefit.",
    "open_or_download_attachment": "Open or download an attachment or file.",
    "install_or_grant_access": "Install software or grant remote access to a device or account.",
    "approve_login_or_transaction": "Approve a login, multi-factor prompt, or transaction.",
    "update_account_or_delivery_details": "Change account, payment, profile, or delivery details.",
    "other_or_unclear": "Another requested action, or the requested action is too vague to tell.",
}

MESSAGE_HOOKS = {
    "account_or_payment_problem": "A blocked account, security alert, failed transaction, or account problem.",
    "delivery_or_order_update": "A parcel, delivery fee, address issue, or order update.",
    "invoice_or_debt": "Money owed: a bill, invoice, debt, fine, or overdue payment; not a bank statement.",
    "prize_refund_or_gift": "A nongovernment prize, refund, compensation, giveaway, or free gift.",
    "government_or_legal_matter": "Government benefits or rebates, taxes, police, courts, or legal matters.",
    "job_or_investment": "A job, paid task, investment, or promised financial return.",
    "tech_support_or_subscription": "A device/security problem or subscription renewal.",
    "relationship_or_emergency": "A personal relationship or family emergency; not routine appointments.",
    "sale_or_promotion": "A sale, product promotion, or ordinary subscription offer.",
    "none_or_other_unclear": "Other premises, routine records or appointments, or no clear hook.",
}

QUESTIONS = {
    "classification": {
        "type": "choice",
        "instructions": (
            "Classify by risk, not claimed authority. Requests for identity or bank details to "
            "release benefits, fees to obtain employment or winnings, unexpected parcel fees, secret "
            "codes, and unsolicited remote access indicate suspected scams. Ordinary discounts are "
            "marketing; routine notices and bills are legitimate."
        ),
        "criteria": CATEGORIES,
    },
    "industry": {
        "type": "choice",
        "instructions": (
            "Which business or service is the message about? Classify the service being provided, not "
            "the method used to pay for it. Identify the claimed industry without verifying identity."
        ),
        "criteria": INDUSTRIES,
    },
    "urgent_action": {
        "type": "noul",
        "instructions": (
            "Does the message pressure the recipient to act soon, meet a deadline, or risk a stated "
            "negative consequence if they delay? Judge the wording, not whether the deadline is "
            "genuine."
        ),
        "criteria": {
            "true": "Yes, the message pressures the recipient to act soon or meet a deadline.",
            "false": "No, it does not pressure the recipient to act soon or meet a deadline.",
        },
        "labels": {"true": "A", "false": "B"},
    },
    "sentiment": {
        "type": "choice",
        "instructions": (
            "What emotional tone does the sender express, regardless of whether the message is "
            "honest?"
        ),
        "criteria": {
            "happy": "Positive, cheerful, celebratory, enthusiastic, or presenting a treat or exciting reward.",
            "neutral": "Factual or matter-of-fact, without clear positive or negative emotional framing.",
            "sad": "Negative, worried, frustrated, threatening, alarming, or distressing tone.",
        },
    },
    "requested_action": {
        "type": "choice",
        "instructions": (
            "What is the main action requested by this message?"
        ),
        "criteria": REQUESTED_ACTIONS,
    },
    "multiple_actions": {
        "type": "noul",
        "instructions": (
            "Does the sender request BOTH of two independent actions, rather than one action or "
            "alternatives? Count different requests, not link steps or multiple data fields. An "
            "explicit opt-out counts; advertising alone does not request purchase."
        ),
        "criteria": {
            "true": "Yes, it asks for two or more distinct actions.",
            "false": "No, it asks for at most one distinct action.",
        },
        "labels": {"true": "A", "false": "B"},
    },
    "message_hook": {
        "type": "choice",
        "instructions": (
            "Identify the main premise attracting attention, regardless of truth. Prefer government "
            "for government benefits. A record or reminder is not a debt or emergency without such a "
            "claim."
        ),
        "criteria": MESSAGE_HOOKS,
    },
    "sensitive_data_requested": {
        "type": "noul",
        "instructions": (
            "Does the message request a password, one-time security code, government identity number, "
            "bank account details, card details, or identity document from the recipient? Routine "
            "sign-in, payment and opt-out instructions alone do not request disclosure."
        ),
        "criteria": {
            "true": "Yes, it asks the recipient to disclose security codes, passwords, bank/card details or government identity data.",
            "false": "No, it does not ask the recipient to disclose those sensitive details.",
        },
        "labels": {"true": "A", "false": "B"},
    },
}
