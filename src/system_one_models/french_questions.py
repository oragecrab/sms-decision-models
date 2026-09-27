"""French model-facing schema; convert outputs back to stable English IDs."""
from copy import deepcopy

from .questions import QUESTIONS

# Same meanings and option order as QUESTIONS; French category names are
# model-facing only. Dataset labels and public answer keys stay unchanged.
INSTRUCTIONS = {
    'classification': "Classez selon le risque, pas l'autorité revendiquée. Demander des données d'identité ou bancaires pour débloquer des prestations, des frais pour obtenir un emploi ou un gain, des frais de colis inattendus, des codes secrets ou un accès à distance non sollicité indique une arnaque présumée. Les réductions ordinaires sont du marketing ; les avis et factures habituels sont légitimes.",
    'industry': "De quel secteur ou service parle le message ? Classez le service fourni, pas son moyen de paiement. Identifiez le secteur revendiqué sans vérifier l'identité.",
    'urgent_action': "Le message pousse-t-il à agir bientôt, à respecter une échéance ou évoque-t-il une conséquence négative en cas de retard ? Jugez la formulation, pas la réalité de l'échéance.",
    'sentiment': "Quel ton émotionnel l'expéditeur exprime-t-il, que le message soit honnête ou non ?",
    'requested_action': "Quelle est l'action principale demandée par ce message ?",
    'multiple_actions': "L'expéditeur demande-t-il deux actions indépendantes ensemble, plutôt qu'une seule action ou des alternatives ? Comptez les demandes distinctes, pas les étapes d'un lien ni les champs de données. Une désinscription explicite compte ; la publicité seule ne demande pas d'achat.",
    'message_hook': "Identifiez le prétexte principal qui attire l'attention, indépendamment de sa véracité. Préférez le gouvernement pour les prestations publiques. Un document ou rappel n'est pas une dette ou urgence sans une telle affirmation.",
    'sensitive_data_requested': "Le message demande-t-il un mot de passe, un code de sécurité à usage unique, un numéro d'identité gouvernemental, des coordonnées bancaires, des données de carte ou une pièce d'identité ? Se connecter, payer ou se désinscrire normalement ne demande pas à lui seul une divulgation.",
}
OPTIONS = {
    'classification': [
        ('Légitime', 'Avis habituels de compte, rappels de rendez-vous, messages personnels ou factures normales de services.'),
        ('Marketing', 'Ventes commerciales, réductions, promotions de produits ou publicité ordinaire.'),
        ('Arnaque présumée', "Hameçonnage ou fraude probable : frais suspects, codes secrets, données personnelles ou financières, accès à distance non sollicité."),
    ],
    'industry': [
        ('Finance', 'Banques, comptes financiers, placements, assurances ou finances personnelles.'),
        ('Commerce', 'Achats, commandes en ligne, places de marché ou produits de consommation.'),
        ('Livraison', 'Livraison de colis, poste, fret ou suivi de messagerie.'),
        ('Services publics', 'Gouvernement, impôts, prestations, eau, électricité ou services publics.'),
        ('Technologie', 'Logiciels, comptes en ligne, appareils, internet ou téléphonie.'),
        ('Santé', 'Soins, pharmacies, services médicaux ou assurance maladie.'),
        ('Emploi', 'Emplois, recrutement, paie ou services professionnels.'),
        ('Voyage', 'Voyages, hébergement, transport ou hôtellerie.'),
        ('Autre', 'Secteur identifiable ne correspondant pas aux catégories proposées.'),
        ('Indéterminé', "Informations insuffisantes pour identifier un secteur."),
    ],
    'sentiment': [
        ('Positif', 'Ton positif, joyeux, festif, enthousiaste, ou présentant un plaisir ou une récompense excitante.'),
        ('Neutre', 'Ton factuel, sans cadrage émotionnel clairement positif ou négatif.'),
        ('Négatif', 'Ton négatif, inquiet, frustré, menaçant, alarmant ou angoissant.'),
    ],
    'requested_action': [
        ('Aucune action', 'Aucune action concrète ; information ou notification seulement.'),
        ('Contact', "Répondre, appeler, contacter, se désinscrire ou poursuivre ailleurs."),
        ('Lien', 'Ouvrir un lien, visiter un site ou scanner un code QR.'),
        ('Connexion', "Se connecter, réinitialiser un mot de passe, vérifier un compte ou une identité."),
        ('Divulgation', 'Fournir mots de passe, codes, données personnelles ou financières, ou documents.'),
        ('Paiement', 'Payer frais, facture, amende ; envoyer argent, cartes-cadeaux ou cryptomonnaie.'),
        ('Achat', "Acheter, souscrire, utiliser une offre ou réclamer un prix, remboursement ou prestation."),
        ('Pièce jointe', 'Ouvrir ou télécharger une pièce jointe ou un fichier.'),
        ('Installation', 'Installer un logiciel ou accorder un accès à distance à un appareil ou compte.'),
        ('Approbation', "Approuver une connexion, une demande multifacteur ou une transaction."),
        ('Modification', 'Modifier les données de compte, paiement, profil ou livraison.'),
        ('Autre ou vague', 'Autre action demandée, ou action trop vague pour la déterminer.'),
    ],
    'message_hook': [
        ('Problème de compte', 'Compte bloqué, alerte de sécurité, transaction échouée ou problème de compte.'),
        ('Colis ou commande', 'Colis, frais de livraison, problème d’adresse ou suivi de commande.'),
        ('Facture ou dette', 'Somme due : facture, dette, amende ou retard de paiement ; pas un relevé bancaire.'),
        ('Prix ou cadeau', 'Prix, remboursement, indemnisation, cadeau ou gratuité non gouvernemental.'),
        ('Gouvernement ou justice', 'Prestations ou remises publiques, impôts, police, tribunaux ou questions juridiques.'),
        ('Emploi ou placement', 'Emploi, tâche rémunérée, placement ou rendement financier promis.'),
        ('Assistance ou abonnement', 'Problème d’appareil ou de sécurité, ou renouvellement d’abonnement.'),
        ('Relation ou urgence', 'Relation personnelle ou urgence familiale ; pas un rendez-vous habituel.'),
        ('Promotion', 'Vente, promotion de produit ou offre ordinaire d’abonnement.'),
        ('Aucun ou autre', 'Autre prétexte, documents ou rendez-vous habituels, ou prétexte indéterminé.'),
    ],
}
BOOLEAN_CRITERIA = {
    'urgent_action': {'true': 'Oui, le message pousse à agir bientôt ou à respecter une échéance.', 'false': 'Non, le message ne pousse pas à agir bientôt ou à respecter une échéance.'},
    'multiple_actions': {'true': 'Oui, au moins deux actions distinctes sont demandées.', 'false': 'Non, au plus une action distincte est demandée.'},
    'sensitive_data_requested': {'true': "Oui, le destinataire doit divulguer codes, mots de passe, données bancaires ou de carte, ou données d’identité gouvernementales.", 'false': 'Non, ces données sensibles ne sont pas demandées.'},
}


def french_schema() -> tuple[dict, dict]:
    schema, mappings = {}, {}
    for key, original in QUESTIONS.items():
        question = deepcopy(original)
        question['instructions'] = INSTRUCTIONS[key]
        if original['type'] == 'choice':
            options = OPTIONS[key]
            canonical = list(original['criteria'])
            if len(options) != len(canonical):
                raise ValueError(f'French option mismatch for {key}')
            question['criteria'] = dict(options)
            mappings[key] = dict(zip((name for name, _ in options), canonical, strict=True))
        else:
            question['criteria'] = BOOLEAN_CRITERIA[key].copy()
            # A/B labels retain semantic true/false slots without English words.
        schema[key] = question
    return schema, mappings


def canonical_answers(raw: dict, mappings: dict) -> dict:
    answer = deepcopy(raw)
    for key, mapping in mappings.items():
        row = answer[key]
        row['choice'] = mapping[row['choice']]
        row['probabilities'] = {mapping[label]: probability for label, probability in row['probabilities'].items()}
        if 'logits' in row and isinstance(row['logits'], dict):
            row['logits'] = {mapping[label]: value for label, value in row['logits'].items()}
    return answer
