import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

# Initialize the Firebase app
cred = credentials.Certificate('parking-77f66-firebase-adminsdk-fbsvc-9bdab7b973.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://parking-77f66-default-rtdb.firebaseio.com/'
})

# Reference to the root of the database
root_ref = db.reference('/slots')
# root_ref.child('slot1').set('0')
# root_ref.child('slot2').set('0')
# root_ref.child('slot3').set('0')
# root_ref.child('slot4').set('0')
# root_ref.child('slot5').set('0')
# root_ref.child('slot6').set('1')
slot_value = root_ref.get()
print(slot_value)
# # Set each slot individually



# def get_slot_value(slot_name):
#     try:
#         # Reference to the specific slot
#         slot_ref = db.reference(f'/parking/{slot_name}')
#         # Retrieve the value of the slot
#         slot_value = slot_ref.get()
#         if slot_value is not None:
#             print(f'{slot_name}: {slot_value}')
#         else:
#             print(f'{slot_name} does not exist in the database.')
#     except Exception as e:
#         print(f'An error occurred: {e}')

# # Example usage
# get_slot_value('slot1')

