#This file incorporates code segments generated through open-source AI software.
#This file incorporates code written by multiple authors including Yehan Subasinghe, Zihan Fang, and Zachary Kargas.
import streamlit as st
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
import easyocr
import re
import math
import statistics
import numpy as np
import difflib
import geopandas as gpd
from shapely.geometry import Point, Polygon
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
reader = easyocr.Reader(['en'])

#the following function checks if the device type is valid and reads in the csv file
@st.cache_data
def input_raw_data(file, device_type):
    if device_type == 'Tobii':
        data = pd.read_csv(file) 
    elif device_type == 'WebGazer':
        data = pd.read_csv(file)
    else:
        raise ValueError("Unsupported device type") #may be necessary
    return data

#the following function creates an array of files from the files uploaded and returns it to the caller
@st.cache_data
def make_list(file_list, device_type):
    all_data = []
    for uploaded_file in file_list:
        data = input_raw_data(uploaded_file, device_type)
        all_data.append(data)
    return all_data

#zach's fifth step
def bounding_box_part5():
    Bounding_Boxes = {} # dictionary that holds geopandas shapes of all bounding boxes
    height = 1080 # height of computer screen
    width = 1920 # width of computer screen
    # normalizing to be between 0-1 
    # tobii eye-tracker calculates gaze point in 0-1
    def normalize_coordinates(x, y, w, h):
        x /= width
        y /= height
        w /= width
        h /= height
        return x, y, w, h
    
    # checks whether bounding box is already in dictionary,
    # otherwise this turns the bounding box coordinates into 
    # a GeoSeries of rectangle shapes.
    # Later, the gaze coordinate will be matched to these boxes.
    def make_shapes(filename):
        name = filename.split("_")[3]
        name = f"{name}"
        # print("filename", filename, name)
    
        if name in Bounding_Boxes:
            return Bounding_Boxes[name]
        else:
            ref = pd.read_csv(f'./word_coordinates_split/{name}')
            boxes = pd.DataFrame()

            for i in range(len(ref)): # creating shapes for each word in the file
                x = float(ref.loc[i, 'tobii_x'])
                y = float(ref.loc[i, 'tobii_y'])
                w = float(ref.loc[i, 'tobii_width'])
                h = float(ref.loc[i, 'tobii_height'])
                word = '{word}.{num}'.format(word=ref.loc[i, 'word'], num=ref.loc[i, 'occurrence'])
                # Creating a shape for each word based on its coordinates
                new_tangle = pd.Series({word: Polygon([(x, y), (x+w, y), (x, y+h), (x+w, y+h)])}) 
                boxes = pd.concat([boxes, new_tangle]) # adding this to a new data structure 
            Bounding_Boxes[name] = boxes
            return boxes
    # In addition to the bounding boxes, here I'm adding 
    # Areas of interest for the reading questions, the participants'
    # summaries, and the code. This is just run once.
    def make_aois():
        aois = [(1175, 200, 700, 125),  # prewritten summary
                (1175, 450, 600, 100),  # 'accurate' question
                (1175, 550, 600, 150),  # 'missing' question
                (1175, 725, 600, 100),  # 'unnecessary' question
                (1175, 825, 600, 125),  # 'readable' question
                (0,    85, 1160, 975),  # code
                (1175, 85,  720, 300)]  # participant summary

        tobii_aois = []
        aoi_names = ['prewritten', 'accurate', 'missing', 'unnecessary', 'readable', 'code']
        writing_aoi_names = ['code', 'participant_summary']

        for box in aois:
            x, y, w, h = normalize_coordinates(box[0], box[1], box[2], box[3])
            tobii_aois.append((x, y, w, h))
        
        reading_boxes = pd.DataFrame()
        for i in range(len(aoi_names)):
            x = tobii_aois[i][0] 
            y = tobii_aois[i][1]
            w = tobii_aois[i][2]
            h = tobii_aois[i][3]
            new_tangle = pd.Series({aoi_names[i]: Polygon([(x, y), (x+w, y), (x, y+h), (x+w, y+h)])})
            reading_boxes = pd.concat([reading_boxes, new_tangle])
    
        writing_boxes = pd.DataFrame()
        for i in range(len(writing_aoi_names)):
            x = tobii_aois[i+5][0]
            y = tobii_aois[i+5][1]
            w = tobii_aois[i+5][2]
            h = tobii_aois[i+5][3]
            new_tangle = pd.Series({writing_aoi_names[i]: Polygon([(x, y), (x+w, y), (x, y+h), (x+w, y+h)])})
            writing_boxes = pd.concat([writing_boxes, new_tangle])
    
        return reading_boxes, writing_boxes
    # takes coordinate for left and right eye,
    # then averages them to get gaze point.
    # Returns a geopandas point
    def get_gaze_point(row):
        gaze_left = row['gaze_left_eye']
        gaze_right = row['gaze_right_eye']
        temp_left = re.split(r'[(,\)]', gaze_left) # tuple is a string, so using regex to get the numbers
        temp_right = re.split(r'[(,\)]', gaze_right)
    
        # averaging right and left gaze points for one gaze point
        gaze_point = (statistics.fmean([float(temp_left[1]), float(temp_right[1])]), 
                    statistics.fmean([float(temp_left[2]), float(temp_right[2])]))
        if pd.isna(gaze_point[0]) or pd.isna(gaze_point[1]): # if either eye is invalid
            return -1
        else:
            return [Point(gaze_point[0], gaze_point[1])]
    # assigning point to a box and/or aoi
    # this is the most computationally expensive part
    def localize_gaze(gaze_point, row, boxes):
        pnt = gpd.GeoDataFrame(geometry=gaze_point)
        # confusing, but basically this below line returns false/true whether the point is in a shape
        pnt = pnt.assign(**{key: pnt.within(geom) for key, geom in boxes.items()}) 
        temp = pd.DataFrame(pnt)
        temp = temp.replace({True : 1, False : ''}) # converting 'True' and 'False' to 1 and nothing
        return pd.concat([row, temp.T]) # adding this true/false information to the original row
    gaze_files = os.listdir("./gaze") #this probably should be based on what's uploaded
    reading_aois, writing_aois = make_aois()
    for file in gaze_files: # this should not be necessary in the future
        temp = re.split("_", file)
        pid = temp[0]
        func = re.sub(".csv", "", temp[-1])
        # print("file", file)
        # path for output
        path = "./annotated_gaze"
        try:
            os.mkdir(path)
        except:
            print("folder already exists")
        
        # # eyetracking files
        # eye_file = open(f"/home/zachkaras/pickle_data/{file}", "rb")
        
        # contents = pickle.load(eye_file)
        all_files = dict() # will store all participant's files as a pkl file

        # for key,values in contents.items(): # iterating through all participant's gaze files
        #     print(key)    
        boxes = make_shapes(file)
        if re.search("reading", file):
            boxes = pd.concat([boxes, reading_aois])
        elif re.search("writing", file):
            boxes = pd.concat([boxes, writing_aois])
        
        boxes = gpd.GeoSeries(boxes[0]) # turning boxes into geopandas object
        
        # df = pd.DataFrame.from_dict(values).T
        df = pd.read_csv(f"gaze/{file}")
        # df.insert(0, '', pid)
        num_cols = len(df.columns)
        
        if num_cols == 13: # older files didn't include data for distance from eye-tracker
            df.columns = ['participant_id', 'function_name', 'function_id', 'system_timestamp',
                                'device_timestamp', 'valid_gaze_left', 'valid_gaze_right', 
                                'gaze_left_eye', 'gaze_right_eye', 'valid_pd_left', 'valid_pd_right',
                                'pd_left', 'pd_right']
        elif num_cols == 17:
            df.columns = ['participant_id', 'function_name', 'function_fid', 'system_timestamp',
                                'device_timestamp', 'valid_gaze_left', 'valid_gaze_right', 
                                'gaze_left_eye', 'gaze_right_eye', 'valid_pd_left', 'valid_pd_right',
                                'pd_left', 'pd_right', 'irl_left_eye_coordinates', 
                                'irl_right_eye_coordinates', 'irl_left_point_on_screen', 
                                'irl_right_point_on_screen']
        else:
            #print(f"weird column length. Participant: {file} | File: {key} | # Columns: {num_cols}")
            continue # the only files without 14 or 18 columns have 0 columns

        # iterate through each file, get gaze point
        new_df = pd.DataFrame()
        for i, row in df.iterrows(): # through each gaze file
            gaze_point = get_gaze_point(row)
            if gaze_point == -1: # NaN values should be filtered, but if there's anything weird
                print(f"NaN value for participant: {file} | Particpant: {pid}")
                continue
            
            new_row = localize_gaze(gaze_point, row, boxes) # assign gaze point to bounding box/aoi
            new_df = pd.concat([new_df, new_row], axis=1)
            
        new_df = new_df.T
        new_df.to_csv(f"{path}/{file}.csv")
        print("Complete!")
        # all_files[key] = new_df.to_dict('records') # dictionary to be stored as a pickle file
        # pickle_dir = f"/home/zachkaras/annotated_pickle/{pid}_all.pkl"
        # with open(pickle_dir, 'wb') as f:
        #     pickle.dump(all_files, f)

#zach's fourth step
def bounding_box_part4():
    box_files = os.listdir("./word_coordinates_preprocessed/")
    # checks whether the word is "null", and makes sure "null" is still put
    # into final file
    def check_word(word):
        if isinstance(word, float) and math.isnan(word):
            return "null"
        return word

    # for each bounding box, calculates the char width to ensure reliable splits
    def calculate_char_width(row):
        return row['width'] / len(row['word'])

    # for words with a dot (e.g. System.out.println), splits and puts each part into a new row
    def split_word(row, char_width):
        parts = re.split("\.", row['word'])
        new_x = row['x']
        replacement = pd.DataFrame()

        for j, string in enumerate(parts):
            word_width = round(len(string)*char_width)
            if word_width == 0:
                continue
            new_row = pd.Series([string, 0, new_x, row['y'], word_width, row['height'], new_x / 1920,
                                row['tobii_y'], word_width / 1920, row['tobii_height']])
            # concatenating new rows together
            replacement = pd.concat([replacement, new_row], ignore_index=True, axis=1)
            new_x += word_width # moving x coordinate by word size
        return replacement

    # occurrence counts for each word got messed up splitting the boxes, so this recalculates for each word
    # this will also discard comments from the calculations
    def recalculate_num_occurrences(new_boxes): 
        occurrences = {}
        comment_line = 0
        slashes = r'\/\/'  # comments in java --> //
    
        for i, row in new_boxes.iterrows():
            word = row['word']
        
            # comment filter: if word is // or a flag is flilpped, just write "comment" in occurrence column
            if re.search(slashes, word) or row['y'] == comment_line:
                comment_line = row['y']
                #occurrences[word] = "comment"
                continue
        
            if word not in occurrences:
                occurrences[str(word)] = 0
            # elif occurrences[word] == 'comment':
            
            else:
                occurrences[str(word)] += 1
            row['occurrence'] = occurrences[word]
        return new_boxes

    # used to find words in code that qualify for splitting
    # 1) if it's a string, 
    # 2) there's a period in it somewhere, and
    # 3) the whole word isn't itself a string
    # e.g. System.out.println but not: "P/x.ctx" because it's in quotes
    def needs_to_be_split(word):
        pattern = r'\"(.+?)\"' # pattern checking whether word is a single string
        return isinstance(word, str) and re.search("\.", word) and not re.findall(pattern, word)

    def process_file(file, new_boxes):
        boxes = pd.read_csv(f'./word_coordinates_preprocessed/{file}') # bounding box file
        for i, row in boxes.iterrows():
            row['word'] = check_word(row['word']) # checks whether word is "null"
        
            if needs_to_be_split(row['word']):
                char_width = calculate_char_width(row)
                replacement = split_word(row, char_width) # splitting word by dots, returning new dataframe rows
                replacement.index = new_boxes.index # currently transposed, so this sets column headers
                new_boxes = pd.concat([new_boxes, replacement], ignore_index=True, axis=1)
            else:
                new_boxes = pd.concat([new_boxes, row.T], ignore_index=True, axis=1)
        file = re.sub("_boxes", "", file)
        new_boxes = recalculate_num_occurrences(new_boxes.T)
        new_boxes.to_csv(f"word_coordinates_split/{file}", index=False, header=[ #this should eventually populate the folder if its not there
            'word', 'occurrence', 'x', 'y', 'width', 'height', 'tobii_x',
            'tobii_y', 'tobii_width', 'tobii_height'])
    for file in box_files:
        new_boxes = pd.DataFrame()
        print(file)
        process_file(file, new_boxes)
    bounding_box_part5()

#zach's third step
def bounding_box_part3(uploaded_file):
    # eye_files = os.listdir('../data/168/gaze/') # eye-tracking file
    box_files = os.listdir('./word_coordinates/') # all bounding boxes
    img_files = os.listdir('./stimuli/') # screenshots of all the stimuli

    # using regular expressions to get function name
    img_names = []
    for i, file in enumerate(box_files):
        img_name = img_files[i].split('.png')[0]
        print(img_name)
        img_names.append(img_name)
    height = 1080
    width = 1920

    def normalize_coordinates(x, y, w, h):
        x /= width
        y /= height
        w /= width
        h /= height
        return x, y, w, h 
    # This for loop normalizes the screen pixels to 0-1 coordinates, 
    # which is how Tobii eyetracker output is formatted
    for i, file in enumerate(box_files):
    
        box_name = re.sub('_boxes.csv', '', box_files[i]) #just the string
        imi = img_names.index(box_name) # index for image corresponding to word coordinates
        curr_img = img_files[imi] # getting the image file name
        img = cv2.imread(str('./stimuli/' + curr_img))
    
        curr_boxes = pd.read_csv(str('./word_coordinates/' + box_files[i])) # reading in word coordinates
        new_df = pd.DataFrame()
        for ii in range(curr_boxes.shape[0]): # adding pixels to align boxes
            x, y, w, h = curr_boxes['x'][ii]+10, curr_boxes['y'][ii]+100, curr_boxes['width'][ii], curr_boxes['height'][ii]
            nx, ny, nw, nh = normalize_coordinates(x, y, w, h) # normalizing here with function
            tobii_coords = pd.Series([nx, ny, nw, nh]).T
            new_row = pd.concat([curr_boxes.iloc[ii, :], tobii_coords])
            new_df = pd.concat([new_df, new_row], axis=1)
        new_df = new_df.T
        # new_df = new_df.drop(['Unnamed: 0'], axis=1)
        new_df.to_csv(str('./word_coordinates_preprocessed/' + box_files[i]), index=False, header=[#this should eventually populate the folder if its not there
            'word', 'predicted_word', 'x', 'y', 'width', 'height', 'tobii_x', 
            'tobii_y', 'tobii_width', 'tobii_height'])
        # This for loop is used to filter out parentheses from the bounding box files
        # And keep track of the occurrences of words in the code (e.g. False.0, False.1, False.2)
        count = 0
        for i, file in enumerate(box_files):
            curr_boxes = pd.read_csv(str('./word_coordinates_preprocessed/' + box_files[i])) # reading in word coordinates
            new_df = pd.DataFrame()
    
            ast_dict = {}
            curr_boxes.insert(loc=1, column="occurrence", value=0)
            for ii in range(curr_boxes.shape[0]): 
                word = curr_boxes['word'][ii]
                pointless = ['(', ')', '()', '{', '}', ');', '),', '));', 
                            '];', '[', ']', '))', '){', ';', ');', '});', '};', '((', ')){', 
                            ')[', '))));', ')))', ')(']
                if word in pointless: # ignoring whole boxes that are parentheses and brackets
                    continue
                elif (isinstance(word, float)):
                    temp = 'null'
                else:
                    # pattern, replace with, string
                    temp = re.sub(r'[(\){\}[\;\,]+$', '', word) # removing parentheses, etc. from the end of strings
            
                if temp not in ast_dict:
                    ast_dict[temp] = 0
                else:
                    ast_dict[temp] += 1

                new_row = curr_boxes.iloc[ii, :]
                new_row['word'] = temp
                new_row["occurrence"] = ast_dict[temp]
                new_df = pd.concat([new_df, new_row], axis=1)
        
            new_df = new_df.T
            new_df = new_df.drop('predicted_word', axis=1)
            new_df.to_csv(str('./word_coordinates_preprocessed/' + box_files[i]), index=False, header=[
                'word', 'occurrence', 'x', 'y', 'width', 'height', 'tobii_x', 
                'tobii_y', 'tobii_width', 'tobii_height'])
            bounding_box_part4()


#zach's second step
def bounding_box_part2():
    oracle = pd.read_csv('./pruned_seeds2.csv')
    box_files = os.listdir('./word_coordinates')

    for boxes in box_files:
        name = re.split(string=boxes, pattern='_boxes.csv')[0]
        idx = np.where(oracle['name'] == name)[0][0]
        function = oracle['function'][idx]
        print(name)
    
        split_string = re.split(string=function, pattern=" |\n")
        split_string = [i for i in split_string if i]
        filename = './word_coordinates/'+boxes
        strings = pd.read_csv('./word_coordinates/'+boxes, index_col=False)
        strings.columns = ['index', 'predicted_word', 'x', 'y', 'width', 'height']
        strings = strings.drop(columns=['index'])
        new_col = pd.DataFrame()
    
        for word in strings['predicted_word']:
            try:
                closest_matches = difflib.get_close_matches(word, split_string, n=3)
                new_col = pd.concat([new_col, pd.Series(closest_matches[0])], ignore_index=True)
            except:
                new_col = pd.concat([new_col, pd.Series('-----------')], ignore_index=True)
        strings['word'] = new_col #.insert(loc=0, column='word', value=new_col)
        strings = strings[['word', 'predicted_word', 'x', 'y', 'width', 'height']]
        strings.to_csv(('./word_coordinates/'+boxes), index=False, header=True)
        

#zach's first step
@st.cache_data
def bounding_box_part1(pic):
    stimdf = pd.read_csv('pruned_seeds2.csv')
    temp_file_path = f"./temp/{pic.name}"
    with open(temp_file_path, "wb") as f:
        f.write(pic.getbuffer())
    #name = re.split('.png', pic)[0]
    name = pic.name.split('.png')[0]
    print(name)
        # temp contains all the images for each word, split by function name
    try:
        os.makedirs(f'./temp/{name}', exist_ok=True)
    except:
        print("file exists")
    boxfile = '{name}_boxes.csv'.format(name=name)
    df = pd.DataFrame()
    row = np.where(stimdf['name'] == name) # finding row specific to each function
    i = row[0][0]
    func = stimdf['function'][i] # getting actual Java code for each stimulus
    split = re.split(" |\n", func) # splitting each function by spaces and newlines to get word level 
    split = list(filter(None, split))
        # now with all the code split up, create actual bounding boxes on images
        # basically just a pipeline with cv2
    
    img = cv2.imread('./stimuli/{pic.name}'.format(pic=pic)) # 
    img = img[100:1000, 10:1150] #make this adjustable
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    dilation = cv2.dilate(thresh, rect_kernel, iterations=1)
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print(len(contours)) # number of words caught by bounding boxes
    unpredicted = 0 # need to do these manually
    for ii, box in enumerate(contours):
        box = contours[(len(contours)-1)-ii]
        x, y, w, h = cv2.boundingRect(box)  # coordinates, width, and height
        tangle = cv2.rectangle(img, (x, y), (x+w, y+h),(0, 255, 0, 2))  # drawing the rectangle
        word_img = img[y+1:(y+1)+(h-1), x+1:(x+1)+(w-1)] # actual pixel values for word
        resized = cv2.resize(word_img, (w*5, h*5),interpolation=cv2.INTER_CUBIC) # bumping up size to improve OCR

        # Convert image to grayscale
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        # Apply thresholding to remove noise
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Perform dilation to make characters more prominent
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        dilated = cv2.dilate(thresh, kernel, iterations=1)
        # We generate a copy of the image and apply a dilation kernel 
        # and median blur on it. - ChatGPT
        blurred = cv2.medianBlur(dilated, 1)
        
        # Perform erosion to remove any remaining noise
        erosion = cv2.erode(blurred, kernel, iterations=1)
        inverted = 255 - erosion

        # Perform OCR on the inverted image
        result = reader.readtext(inverted)
        
        if not result: # sometimes word was null
            cv2.imwrite("./temp/{func}/{c}_unknown.png".format(func=name, c=unpredicted), inverted)
            temp = pd.DataFrame([['unknown_{i}'.format(i=unpredicted), x, y, w, h]]) # still adding the row
            df = pd.concat([df, temp], ignore_index=True)
            unpredicted += 1
        for r in result:
            cv2.imwrite("./temp/{func}/{c}.png".format(c=r[1],func=name), inverted)
            temp = pd.DataFrame([[re.sub(" ", "", r[1]), x, y, w, h]])
            df = pd.concat([df, temp], ignore_index=True)

    cv2.imwrite("./temp/{name}/{c}_func.png".format(c=name, name=name), img)
    df.columns = ['word', 'x', 'y', 'width', 'height']
    
    df = df.sort_values(['y','x']) # some characters had different heights (B vs. +), so the below code 
    count = 0                      # standardizes row values and sorts each row
    standard = df.iloc[0, 2]       # now we have bounding box files that read sequentially in order
    for i, row in df.iterrows():
        if i < len(df)-1:
            diff = df.iloc[i+1, 2]-row[2]
            if diff > 20: # heuristic for new row height
                standard = df.iloc[i+1, 2]
            else:
                row[2] = standard
                df.iloc[i+1, 2] = standard
    df = df.sort_values(['y', 'x'])
    pd.DataFrame.to_csv(df, "./word_coordinates/{name}_boxes.csv".format(name=name)) #this should eventually populate the folder if its not there
    bounding_box_part2()

def home_page():
    st.markdown("<h1 style='text-align: center;'>VisualAnchor</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Welcome to the home page! Proceed to step 1 using the navigation on the left.</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>If files are already separated by participant, skip step 1 and proceed to step 2.</p>", unsafe_allow_html=True)

def step_1():
    st.title("STEP 1")
    st.write("Step 1 content goes here.")

def step_2():
    st.title("STEP 2")
    #creates UI for uploading files
    uploaded_file = st.file_uploader("Upload your eye-tracking data files", accept_multiple_files=True, type=['csv'])
    device_type = st.selectbox("Select device type", ['Tobii', 'WebGazer']) 
    image_path = st.file_uploader("Upload images to overlay bounding boxes", accept_multiple_files=False, type=['png', 'jpg', 'jpeg'])

    if st.button('Process Data'):
        if uploaded_file and image_path: 
            combined_data = make_list(uploaded_file, device_type)
            for file in combined_data: #this will change eventually to restart afterwards
                bounding_box_part1(image_path)
                #ask about how to get this to stop reloading
                name = image_path.name.split('.png')[0]
                output_image = cv2.imread("./temp/{name}/{c}_func.png".format(c=name, name=name))
                st.image(output_image, use_column_width=True)
                byte_img = cv2.imencode(".png", output_image)[1].tobytes()
                st.download_button(
                    label="Download image as PNG",
                    data=byte_img,
                    file_name="bounding_box.png",
                    mime="bounding_box/png",
                )
                st.write("Download this image to view later, then manually correct the file below before moving on to step 3.")
                csv_dir = "./word_coordinates"
                csv_file = "{name}_boxes.csv".format(name=name)
                df = pd.read_csv(os.path.join(csv_dir, csv_file))
                st.title("CSV File Editor")
                edited_df = st.data_editor(df, use_container_width=True)
                if st.button("Save and Download"):
                    # Save the updated DataFrame to the CSV file
                    edited_df.to_csv(os.path.join(csv_dir, csv_file), index=False)
                    st.success("CSV file updated and downloaded.")

def step_3():
    st.title("STEP 3")
    st.write("This step may take several minutes.")
    uploaded_file = st.file_uploader("Upload your corrected data files", accept_multiple_files=True, type=['csv'])
    if st.button('Process Data'):
        if uploaded_file: 
            bounding_box_part3(uploaded_file)
        else:
            st.error("Please upload corrected file")

def parse_coordinates(coord_str):
    match = re.match(r"\(([-+]?[0-9]*\.?[0-9]+),\s*([-+]?[0-9]*\.?[0-9]+)\)", coord_str)
    if match:
        return float(match.group(1)), float(match.group(2))
    else:
        return None, None

def scanpath():

    st.title('Eye Tracking Scanpath Overlay')

    # File uploaders
    uploaded_image = st.file_uploader("Choose an image file", type=["png"])
    uploaded_csv = st.file_uploader("Choose an eye-tracking data CSV file", type=["csv"])

    if uploaded_image and uploaded_csv:
        # Read the image
        image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Read the CSV file
        eye_tracking_data = pd.read_csv(uploaded_csv)

        if 'gaze_left_eye' in eye_tracking_data.columns and 'gaze_right_eye' in eye_tracking_data.columns:
            left_x_coords, left_y_coords = [], []
            right_x_coords, right_y_coords = [], []

            for left_eye, right_eye in zip(eye_tracking_data['gaze_left_eye'], eye_tracking_data['gaze_right_eye']):
                left_x, left_y = parse_coordinates(left_eye)
                right_x, right_y = parse_coordinates(right_eye)

                left_x_coords.append(left_x)
                left_y_coords.append(left_y)
                right_x_coords.append(right_x)
                right_y_coords.append(right_y)
            
            # Plot the scanpath on the image
            fig, ax = plt.subplots()
            ax.imshow(image)

            # Plot left eye gaze
            ax.plot(left_x_coords, left_y_coords, marker='o', color='red', linestyle='-', label='Left Eye')
            # Plot right eye gaze
            ax.plot(right_x_coords, right_y_coords, marker='o', color='blue', linestyle='-', label='Right Eye')

            # Add legend
            ax.legend()

            # Display the plot
            st.pyplot(fig)
        else:
            st.error("CSV file must contain 'x' and 'y' columns.")
    else:
        st.info("Please upload both an image and a CSV file.")

def heatmap():
    st.title('Eye Tracking Heatmap Overlay')

    # File uploaders
    uploaded_image = st.file_uploader("Choose an image file", type=["png"])
    uploaded_csv = st.file_uploader("Choose an eye-tracking data CSV file", type=["csv"])

    if uploaded_image and uploaded_csv:
        # Read the image
        image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Read the CSV file
        eye_tracking_data = pd.read_csv(uploaded_csv)

        if 'gaze_left_eye' in eye_tracking_data.columns and 'gaze_right_eye' in eye_tracking_data.columns:
            gaze_points = []

            for left_eye, right_eye in zip(eye_tracking_data['gaze_left_eye'], eye_tracking_data['gaze_right_eye']):
                left_x, left_y = parse_coordinates(left_eye)
                right_x, right_y = parse_coordinates(right_eye)

                if left_x is not None and left_y is not None:
                    gaze_points.append((left_x, left_y))
                if right_x is not None and right_y is not None:
                    gaze_points.append((right_x, right_y))

                # Convert gaze points to numpy array
                gaze_points = np.array(gaze_points)

                # Create a heatmap
                heatmap, xedges, yedges = np.histogram2d(gaze_points[:, 0], gaze_points[:, 1], bins=100)
                heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

                # Normalize the heatmap
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
                heatmap = np.uint8(255 * heatmap)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

                # Overlay the heatmap on the image
                overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

                # Display the image with heatmap overlay
                st.image(overlay, caption='Heatmap Overlay', use_column_width=True)
        else:
            st.error("CSV file must contain 'left_eye' and 'right_eye' columns.")
    else:
        st.info("Please upload both an image and a CSV file.")


def main():
    if 'page' not in st.session_state:
        st.session_state.page = 'Home'
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", ["Home", "Step 1", "Step 2", "Step 3", "Scanpath", "Heatmap"], index=["Home", "Step 1", "Step 2", "Step 3", "Scanpath", "Heatmap"].index(st.session_state.page), key='sidebar_radio')
    if selection != st.session_state.page:
        st.session_state.page = selection
    # Display the selected page
    if st.session_state.page == "Home":
        print("Home")
        home_page()
    elif st.session_state.page == "Step 1":
        print("Step 1")
        step_1()
    elif st.session_state.page == "Step 2":
        print("Step 2")
        step_2()
    elif st.session_state.page == "Step 3":
        print("Step 3")
        step_3()
    elif st.session_state.page == "Scanpath":
        print("Scan")
        scanpath()
    elif st.session_state.page == "Heatmap":
        heatmap()

if __name__ == "__main__":
    main()