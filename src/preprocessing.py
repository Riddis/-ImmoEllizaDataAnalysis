from jsonschema import validate
from jsonschema import Draft202012Validator
import pandas as pd
from sklearn.externals import joblib

def preprocess_new_data(json_data):
    
    json_schema = {
        "type" : "object",
        "properties" : {
            "data" : {
                "type" : "object",
                "properties" : {
                    "area": {type: "integer"},
                    "property-type" : {
                        type: "string", 
                        "enum": ['APARTMENT', 
                                'HOUSE']
                    },
                    "rooms-number": {type: "integer"},
                    "zip-code": {type: "integer"},
                    "full-address": {type: "string"},
                    "land-area": {type: "integer"},
                    "garden": {type: "boolean"},
                    "garden-area": {type: "integer"},
                    "equipped-kitchen": {type: "string"},
                    "full-address": {type: "string"},
                    "swimming-pool": {type: "boolean"},
                    "furnished": {type: "boolean"},
                    "open-fire": {type: "boolean"},
                    "terrace": {type: "boolean"},
                    "terrace-area": {type: "integer"},
                    "facades-number": {type: "integer"},
                    "building-state": {
                        type: "string",
                        "enum": ['NEW', 
                                'GOOD', 
                                'RENOVATE', 
                                'JUST RENOVATED', 
                                'TO REBUILD']
                    },
                }, 
                "required": ["area", 
                             "property-type", 
                             "rooms-number", 
                             "zip-code"],
                "additionalProperties": False
            }
        },
        "required": ["data"],
        "additionalProperties": False
    }

    if validate(instance=json_data, schema=json_schema) == False:
        v = Draft202012Validator(json_schema)
        errors = sorted(v.iter_errors(json_data), key=str)
        status = 400

        return errors, status
    
    df = pd.read_json(json_data)
    df = pd.data


    'number_rooms', 'living_area',
    'terrace', 'terrace_area', 'garden',
    'garden_area', 'surface_land', 'number_facades',
    'property_type', 'building_state', 'kitchen', 'province', 'digit'

    encoder = joblib.load("models/encoder.save")
    scaler = joblib.load("models/scaler.save") 

    X = pd.to_numpy()
    X = encoder.transform(df)
    X = scaler.transform(X)

    status = 200
    return X, status




 # model attributes: feature names


'''from sklearn.externals import joblib

# Save
encoder_filename = "models/encoder.save"
joblib.dump(encoder, encoder_filename)
# Load
encoder = joblib.load("models/encoder.save")









    property_types = ['APARTMENT', 'HOUSE']
    if jsonData['property_type'] not in property_types:
        raise ValueError("Invalid property type. Expected one of: %s" % property_types)
    
    provinces = ['Antwerp', 'Brussels', 'East Flanders', 
                     'West Flanders', 'Flemish Brabant', 'Hainaut',
                     'Liège', 'Limburg', 'Luxembourg', 'Namur', 'Waloon Brabant']
    if province not in provinces:
        provinces = list(filter(lambda x: x is not None, provinces))
        raise ValueError("Invalid kitchen type. Expected one of: %s" % provinces)
    
    kitchen_types = ['NOT_INSTALLED', 'USA_UNINSTALLED', 'SEMI_EQUIPPED', 
                     'USA_SEMI_EQUIPPED', 'INSTALLED', 'USA_INSTALLED',
                     'HYPER_EQUIPPED', 'USA_HYPER_EQUIPPED', None]
    if kitchen not in kitchen_types:
        kitchen_types = list(filter(lambda x: x is not None, kitchen_types))
        raise ValueError("Invalid kitchen type. Expected one of: %s" % kitchen_types)

    building_states = ['NEW', 'GOOD', 'RENOVATE', 
                       'JUST RENOVATED', 'TO REBUILD', None]
    if building_state not in building_states:
        building_states = list(filter(lambda x: x is not None, building_states))
        raise ValueError("Invalid building state. Expected one of: %s" % building_states)
    
    digit = str(int(digit)/100)

    arguments = locals()
    dict = {'data' : ''}
    
    for key, value in arguments.items():
        dict['data'][key] = value

    json_obj = json.dumps(dict)
    
    return json_obj


"""(living_area:int, 
                        property_type:str, 
                        number_rooms:int, 
                        digit:int, 
                        province:str = None,
                        surface_land:int = None,
                        garden:bool = None, 
                        garden_area:int = None,
                        kitchen:str = None,
                        terrace:bool = None, 
                        terrace_area:bool = None,
                        number_facades:int = None,
                        building_state:str = None)"""'''