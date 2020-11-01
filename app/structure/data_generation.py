import pandas as pd
import random as random
import json
import matplotlib.pyplot as plt


#############################################################
# Read Input Json File
#############################################################
# Opening JSON file
f = open('1_5_dynamic_data.json', )

# returns JSON object as
# a dictionary
data = json.load(f)

#Iterating through the json
#list
# for i in data['data']:
#     print(i)

# Closing file
f.close()


# print("Full File Prop: " + str(data["data"]))

first_row = data["data"][0:2]
print("single Object "+str(first_row))

# Get Property value from JSON Array
# RP_benthic_habitats = (first_row["RP_benthic_habitats"])
# print("RP_benthic_habitats : " + str(RP_benthic_habitats))

# Make an array from Json Object
# first_row_array = [first_row["RP_benthic_habitats"], first_row["RP_shipping_container_2016"], first_row["RP_shipping_rorocargo_2016"]]
# plt.plot(first_row_array)
# plt.show()



#############################################################
# New Data Generation - Model 5
#############################################################

# Total Number
total_num_of_mock_data = 20

# Combined Data
result_data = []

# Iterate x times
x = range(total_num_of_mock_data)
print("Range : "+ str(x))

for n in x:
    # Business Rules application (Lower , Upper)
    RP_benthic_habitats = random.uniform(105, 190)
    RP_shipping_container_2016 = random.uniform(105, 190)
    RP_shipping_rorocargo_2016 = random.uniform(105, 190)
    RP_shipping_service_2016 = random.uniform(105, 190)
    RP_seabed_slope = random.uniform(105, 190)
    RP_depth = random.uniform(105, 190)
    OP_sediment_cover = random.uniform(105, 190)
    OP_bio_cover = random.uniform(105, 190)

    lst = [RP_benthic_habitats, RP_shipping_container_2016, RP_shipping_rorocargo_2016, RP_shipping_service_2016, RP_seabed_slope, RP_depth, OP_sediment_cover]
    print("Row " + str(lst))
    result_data.append(lst)

print(result_data)
# Calling DataFrame constructor on list
df = pd.DataFrame(result_data)

#print(df)
