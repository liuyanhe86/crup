COARSE_FEW_NERD = ['event', 'person', 'art', 'building', 'location', 'organization', 'product', 'other']
FINE_FEW_NERD = ['art-film', 'product-train', 'other-livingthing', 'art-writtenart', 'building-theater', 'person-actor', 'organization-sportsleague', 'person-soldier', 'other-god', 'location-island', 'event-attack,battle,war,militaryconflict', 'organization-media,newspaper', 'building-restaurant', 'other-astronomything', 'organization-government,governmentagency', 'location-GPE', 'event-sportsevent', 'building-airport', 'product-car', 'other-disease', 'location-mountain', 'product-ship', 'building-other', 'building-hotel', 'organization-showorganization', 'organization-politicalparty', 'event-other', 'building-sportsfacility', 'product-food', 'other-medical', 'organization-sportsteam', 'person-other', 'location-road,railway,highway,transit', 'other-award', 'other-chemicalthing', 'other-law', 'art-music', 'product-other', 'other-language', 'location-bodiesofwater', 'organization-other', 'product-airplane', 'building-library', 'organization-religion', 'art-broadcastprogram', 'person-scholar', 'person-athlete', 'other-currency', 'location-other', 'person-director', 'product-game', 'building-hospital', 'event-protest', 'event-election', 'other-educationaldegree', 'art-other', 'art-painting', 'organization-company', 'product-weapon', 'event-disaster', 'organization-education', 'product-software', 'person-artist,author', 'location-park', 'person-politician', 'other-biologything']
STACKOVERFLOW = ['HTML_XML_Tag', 'Data_Structure', 'File_Name', 'Language', 'Device', 'Value', 'File_Type', 'Version', 'Data_Type', 'Variable_Name', 'Library_Class', 'Library_Variable', 'Class_Name', 'Code_Block', 'User_Interface_Element', 'Application', 'Library_Function', 'Library', 'Operating_System']

def get_ci_coarse_few_nerd():
    return {key: 'data/few-nerd/coarse/disjoint/' + key for key in COARSE_FEW_NERD}

def get_ci_fine_few_nerd():
    return {key: 'data/few-nerd/fine/disjoint/' + key for key in COARSE_FEW_NERD}

def get_ci_stackoverflow():
    return {key: 'data/stackoverlfow/disjoint/' + key for key in STACKOVERFLOW}

def get_online_coarse_few_nerd():
    return {key: 'data/few-nerd/coarse/joint/' + key for key in COARSE_FEW_NERD}

def get_online_fine_few_nerd():
    return {key: 'data/few-nerd/fine/joint/' + key for key in FINE_FEW_NERD}

def get_online_stackoverflow():
    return {key: 'data/stackoverlfow/joint/' + key for key in STACKOVERFLOW}

PROTOCOLS = {
    'sup coarse-few-nerd': 'data/few-nerd/coarse/supervised',
    'sup fine-few-nerd': 'data/few-nerd/fine/supervised',
    'sup stackoverflow': 'data/stackoverflow/supervised',
    'CI coarse-few-nerd': get_ci_coarse_few_nerd(),
    'CI fine-few-nerd': get_ci_fine_few_nerd(),
    'CI stackoverflow': get_ci_stackoverflow(),
    'online coarse-few-nerd': get_online_coarse_few_nerd(),
    'online fine-few-nerd': get_online_fine_few_nerd(),
    'online stackoverflow': get_online_stackoverflow(),
    'multi-task coarse-few-nerd': get_ci_coarse_few_nerd(),
    'multi-task fine-few-nerd': get_ci_fine_few_nerd(),
    'multi-task stackoverflow': get_ci_stackoverflow()
}
