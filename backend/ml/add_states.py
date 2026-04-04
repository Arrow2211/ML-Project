import os
import re

state_mapping = {
    'Delhi': 'Delhi',
    'Amritsar': 'Punjab', 'Jalandhar': 'Punjab', 'Ludhiana': 'Punjab', 'Patiala': 'Punjab', 'Bathinda': 'Punjab',
    'Chandigarh': 'Chandigarh',
    'Shimla': 'Himachal Pradesh', 'Dharamshala': 'Himachal Pradesh',
    'Dehradun': 'Uttarakhand', 'Rishikesh': 'Uttarakhand', 'Mussoorie': 'Uttarakhand', 'Nainital': 'Uttarakhand',
    'Srinagar': 'Jammu & Kashmir', 'Jammu': 'Jammu & Kashmir',
    'Leh': 'Ladakh', 'Kargil': 'Ladakh',
    'Jaipur': 'Rajasthan', 'Jodhpur': 'Rajasthan', 'Udaipur': 'Rajasthan', 'Jaisalmer': 'Rajasthan', 'Bikaner': 'Rajasthan', 'Kota': 'Rajasthan', 'Ajmer': 'Rajasthan',
    'Ahmedabad': 'Gujarat', 'Surat': 'Gujarat', 'Rajkot': 'Gujarat', 'Vadodara': 'Gujarat',
    'Bhopal': 'Madhya Pradesh', 'Indore': 'Madhya Pradesh', 'Gwalior': 'Madhya Pradesh', 'Jabalpur': 'Madhya Pradesh', 'Ujjain': 'Madhya Pradesh', 'Sagar': 'Madhya Pradesh', 'Rewa': 'Madhya Pradesh',
    'Lucknow': 'Uttar Pradesh', 'Varanasi': 'Uttar Pradesh', 'Agra': 'Uttar Pradesh', 'Kanpur': 'Uttar Pradesh',
    'Patna': 'Bihar',
    'Ranchi': 'Jharkhand',
    'Raipur': 'Chhattisgarh',
    'Kolkata': 'West Bengal', 'Howrah': 'West Bengal', 'Darjeeling': 'West Bengal', 'Siliguri': 'West Bengal', 'Asansol': 'West Bengal', 'Durgapur': 'West Bengal',
    'Bhubaneswar': 'Odisha', 'Cuttack': 'Odisha',
    'Guwahati': 'Assam', 'Dibrugarh': 'Assam', 'Silchar': 'Assam',
    'Shillong': 'Meghalaya',
    'Imphal': 'Manipur',
    'Gangtok': 'Sikkim',
    'Hyderabad': 'Telangana',
    'Bangalore': 'Karnataka', 'Mysore': 'Karnataka', 'Mangalore': 'Karnataka', 'Hubli': 'Karnataka', 'Belagavi': 'Karnataka', 'Davanagere': 'Karnataka',
    'Chennai': 'Tamil Nadu', 'Coimbatore': 'Tamil Nadu', 'Madurai': 'Tamil Nadu', 'Tiruchirappalli': 'Tamil Nadu', 'Salem': 'Tamil Nadu', 'Erode': 'Tamil Nadu', 'Tirunelveli': 'Tamil Nadu', 'Vellore': 'Tamil Nadu',
    'Thiruvananthapuram': 'Kerala', 'Kochi': 'Kerala', 'Kozhikode': 'Kerala', 'Kollam': 'Kerala', 'Thrissur': 'Kerala',
    'Goa': 'Goa',
    'Visakhapatnam': 'Andhra Pradesh', 'Vijayawada': 'Andhra Pradesh',
    'Pondicherry': 'Puducherry',
    'Port Blair': 'Andaman & Nicobar Islands',
}

file_path = r'c:\Users\athar\OneDrive\Desktop\ML Project\backend\ml\data_fetcher.py'

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
in_array = False
for line in lines:
    if line.strip() == 'INDIAN_CITIES = [':
        in_array = True
        new_lines.append(line)
        continue
        
    if in_array and line.strip() == ']':
        in_array = False
        new_lines.append(line)
        continue
        
    if in_array and '{"city":' in line:
        # Extract city name
        match = re.search(r'"city":\s*"([^"]+)"', line)
        if match:
            city_name = match.group(1)
            state = state_mapping.get(city_name, 'Maharashtra') # default to Maharashtra since all others are covered
            if '"state"' not in line:
                # insert state before zone
                line = line.replace('"zone":', f'"state": "{state}", "zone":')
    new_lines.append(line)

with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("Added state mapping to data_fetcher.py!")
