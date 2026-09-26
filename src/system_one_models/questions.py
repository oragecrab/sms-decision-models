"""Scam-triage question definitions and closed-choice categories."""

CATEGORIES = {
    "legitimate": "A genuine, expected message that is not trying to deceive the recipient.",
    "marketing": "Unsolicited advertising or promotion without an attempt to steal money or information.",
    "suspected_scam": "A deceptive message seeking money, credentials, personal information, or a risky action.",
}

INDUSTRIES = {
    "banking_finance": "Banking, payments, investing, insurance, or personal finance.",
    "retail_ecommerce": "Shopping, online orders, marketplaces, or consumer products.",
    "shipping_logistics": "Parcel delivery, postal services, freight, or courier tracking.",
    "government_public_services": "Government agencies, taxes, benefits, utilities, or public services.",
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
    "share_sensitive_info_or_documents": (
        "Provide passwords, security codes, personal or financial details, or documents."
    ),
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
    "invoice_or_debt": "A bill, invoice, debt, fine, or overdue payment.",
    "prize_refund_or_gift": "A prize, refund, compensation, giveaway, or free gift.",
    "government_or_legal_matter": "A government, tax, police, court, or legal matter.",
    "job_or_investment": "A job, paid task, investment, or promised financial return.",
    "tech_support_or_subscription": "A device/security problem or subscription renewal.",
    "relationship_or_emergency": "A personal relationship, family emergency, or urgent appeal.",
    "sale_or_promotion": "A sale, product promotion, or ordinary subscription offer.",
    "none_or_other_unclear": "No clear hook, or a premise outside these categories.",
}

QUESTIONS = {
    "classification": {
        "type": "choice",
        "instructions": (
            "Classify this SMS or email using the message content and any supplied sender details. "
            "Treat urgency, unexpected payment requests, credential requests, suspicious links, "
            "impersonation, and requests to move a conversation as possible scam indicators. "
            "Do not assume a message is a scam solely because it contains a link."
        ),
        "criteria": CATEGORIES,
    },
    "industry": {
        "type": "choice",
        "instructions": (
            "Identify the claimed industry from the text; do not verify sender identity."
        ),
        "criteria": INDUSTRIES,
    },
    "urgent_action": {
        "type": "noul",
        "instructions": (
            "Does the message pressure the recipient to act soon, meet a deadline, or risk a stated "
            "negative consequence if they delay? Judge the wording, not whether the deadline is genuine."
        ),
        "criteria": {
            "true": "Yes, the message pressures the recipient to act soon or meet a deadline.",
            "false": "No, it does not pressure the recipient to act soon or meet a deadline.",
        },
        "labels": {"true": "A", "false": "B"},
    },
    "sentiment": {
        "type": "choice",
        "instructions": "What overall emotional tone does the sender express in this message?",
        "criteria": {
            "happy": "Positive, cheerful, pleased, or enthusiastic tone.",
            "neutral": "Mostly factual or matter-of-fact, without a clear positive or negative tone.",
            "sad": "Negative, worried, sad, frustrated, or distressing tone.",
        },
    },
    "requested_action": {
        "type": "choice",
        "instructions": (
            "Choose the most consequential requested end goal, not a navigation step. "
            "Payment via a link is pay_or_transfer. For independent actions, choose the riskiest. "
            "Informational messages are no_action. Ignore whether claims are true."
        ),
        "criteria": REQUESTED_ACTIONS,
    },
    "multiple_actions": {
        "type": "noul",
        "instructions": (
            "Does the message explicitly ask the recipient to perform two or more distinct actions? "
            "Do not count a link as a separate action when it only enables the requested sign-in, "
            "payment, or other end goal."
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
            "Identify the main attention hook, regardless of truth. "
            "Use none_or_other_unclear when no option fits."
        ),
        "criteria": MESSAGE_HOOKS,
    },
    "sensitive_data_requested": {
        "type": "noul",
        "instructions": (
            "Does the message ask the recipient to disclose a password, one-time or recovery "
            "security code, bank/card details, government identifier, or identity document? "
            "Do not count an opt-out reply such as STOP or a routine sign-in instruction by itself."
        ),
        "criteria": {
            "true": "Yes, it requests a password, security code, financial detail, or identity data/document.",
            "false": "No, it does not request those details; opt-out replies and routine sign-in alone do not count.",
        },
        "labels": {"true": "A", "false": "B"},
    },
}
