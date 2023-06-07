import cv2
import os
import pandas as pd
from scriptTerminalInput import ColorPaletteExtractor
import find_closest

# Specify the path to the folder containing the images
folder_path = "/Users/irene/Desktop/YoonaRepo/ColorExtractorIre/colorExtYoona"

 # Creazione del DataFrame
df = pd.DataFrame(columns=[
    'filename',
    'Color name 1',
    'RGB code 1',
    'Hex code 1',
    'Pantone code 1',
    'Cmyk code 1',
    'Color name 2',
    'RGB code 2',
    'Hex code 2',
    'Pantone code 2',
    'Cmyk code 2',
    'Color name 3',
    'RGB code 3',
    'Hex code 3',
    'Pantone code 3',
    'Cmyk code 3',
    'Color name 4',
    'RGB code 4',
    'Hex code 4',
    'Pantone code 4',
    'Cmyk code 4',
    'Color name 5',
    'RGB code 5',
    'Hex code 5',
    'Pantone code 5',
    'Cmyk code 5'
])

# Iterate over all the files in the folder
for filename in os.listdir(folder_path):

    # Check if the file is an image
    if filename.endswith((".jpg", ".jpeg", ".png")):

        # Construct the full path to the image file
        file_path = os.path.join(folder_path, filename)
        
        # Load image using OpenCV
        image = cv2.imread(file_path)

        # convert to RGB from BGR
        img_c = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        low_cutoff = 0.0
        high_cutoff = 1.0
        color_mode = "RGB"
        use_agglomerative = False
        
        for c in range(1, 6):
            # Create ColorPaletteExtractor object with default parameters
            cpe =  ColorPaletteExtractor(img_c, c, low_cutoff, high_cutoff, color_mode, agglomerative_clustering=use_agglomerative)
        
            # Extract dominant colors from image
            colors = cpe.extract_colors()

        row_values = {}

        # Inserting color extraction results into dataframe
        for i, color in enumerate(colors):

            color_name = color[2]
            rgb_code = color[1]
            hex_code = color[0]
            pantone_code = find_closest.rgb_to_pantone(*color[1])
            cmyk_code = find_closest.rgb_to_cmyk(*color[1])
            
            column_index = i + 1 
            row_values[f'Color name {column_index}'] = color_name
            row_values[f'RGB code {column_index}'] = rgb_code
            row_values[f'Hex code {column_index}'] = hex_code
            row_values[f'Pantone code {column_index}'] = pantone_code
            row_values[f'Cmyk code {column_index}'] = cmyk_code

        df = df.append({'filename': filename, **row_values}, ignore_index=True)
        
print(df)

# Export dataframe in a csv
df.to_csv('output.csv', index=False)

