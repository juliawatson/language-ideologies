import collections


# MODEL_NAME = "flan-t5-small"
# MODEL_NAME = "flan-t5-large"
# MODEL_NAME = "flan-t5-xl"

# MODEL_NAME = "text-davinci-003"
# MODEL_NAME = "text-davinci-002"
# MODEL_NAME = "text-curie-001"

# MODEL_NAME = "llama-3-8B"
# MODEL_NAME = "llama-3.1-8B"
# MODEL_NAME = "llama-2-7B"

### NAMES CONSTANTS ###

# camilliere_names = {
#     "gendered" : {"Aaron", "Adeline", "Alice", "Amanda", "Amelia", "Annabella", "Bella", "Brandon", "Bridget", "Caleb",
#                     "Charlotte", "Daniel", "David", "Elena", "Elizabeth", "Ella", "Emily", "Emma", "Gianna", "Grant", "Haley", 
#                     "Henry", "Isaac", "Jacob", "John", "Joshua", "Justin", "Lily", "Lucas", "Maria", "Mary", "Molly", 
#                     "Nicholas", "Penelope", "Robert", "Scarlett", "Vivian", "Wyatt", "Zach", "Zoey"}, 
#     "non-gendered" : {"Alex", "Cameron", "Casey", "Dakota", "Finley", "Frankie", "Harper", "Hayden", "Jayden", "Jordan", "Justice", 
#                       "Landry", "Leighton", "Marley", "Morgan", "Pat", "Payton", "Remi", "Sammy", "Skyler","Taylor"}
# }


### ROLE NOUNS CONSTANTS ### 

# ROLE_NOUNS = [
#     ('anchor', 'anchorman', 'anchorwoman'),
#     ('flight attendant', 'steward', 'stewardess'),
#     ('businessperson', 'businessman', 'businesswoman'),
#     ('camera operator', 'cameraman', 'camerawoman'),
#     ('congressperson', 'congressman', 'congresswoman'),
#     ('craftsperson', 'craftsman', 'craftswoman'),
#     ('crewmember', 'crewman', 'crewwoman'),
#     ('firefighter', 'fireman', 'firewoman'),
#     ('foreperson', 'foreman', 'forewoman'),
#     ('layperson', 'layman', 'laywoman'),
#     ('police officer', 'policeman', 'policewoman'),
#     ('salesperson', 'salesman', 'saleswoman'),
#     ('stunt double', 'stuntman', 'stuntwoman'),
#     ('meteorologist', 'weatherman', 'weatherwoman')
# ]

# ROLE_NOUN_REFORM_VARIANTS = [item[0] for item in ROLE_NOUNS]

AN_NOUNS = {
    ('anchor', 'anchorman', 'anchorwoman'),
    ('emergency medical technician', 'ambulanceman', 'ambulancewoman'),
    ('anchorperson', 'anchorman', 'anchorwoman'),
    ('outdoorsperson', 'outdoorsman', 'outdoorswoman'),
    ('alderperson', 'alderman', 'alderwoman'),
    ('ombudsperson', 'ombudsman', 'ombudswoman'),
    ('assemblyperson', 'assemblyman', 'assemblywoman')
}
