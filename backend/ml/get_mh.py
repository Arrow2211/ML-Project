import urllib.request, json, os

url = 'https://raw.githubusercontent.com/nshntarora/Indian-Cities-JSON/master/cities.json'
try:
    with urllib.request.urlopen(url) as response:
        cities = json.loads(response.read().decode())
        
    mh_cities = [c for c in cities if c.get('state') == 'Maharashtra']
    
    new_cities_str = ''
    for c in mh_cities:
        city_name = c['name']
        lat = float(c['lat'])
        lon = float(c['lng'])
        zone = 'coastal_west' if lon < 73.5 else ('inland_west' if lon < 76.0 else 'inland_central')
        new_cities_str += f'    {{"city": "{city_name}", "lat": {lat}, "lon": {lon}, "zone": "{zone}"}},\n'
    
    # Path to data_fetcher
    file_path = r'c:\Users\athar\OneDrive\Desktop\ML Project\backend\ml\data_fetcher.py'
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    insert_marker = ']\n\n# Cache directory'
    if insert_marker in content:
        content = content.replace(insert_marker, new_cities_str + insert_marker)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f'SUCCESS: Added {len(mh_cities)} Maharashtra districts and talukas.')
    else:
        print('ERROR: Could not find insert marker in data_fetcher.py')
except Exception as e:
    print('ERROR:', e)
