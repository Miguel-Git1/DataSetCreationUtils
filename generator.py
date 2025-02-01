from pathlib import Path
import random
import PIL
from  PIL import Image
import cv2 
import numpy as np
import json

class DatasetBuilder():
    '''The builder will first take the background, it can be a file or a directory\n
       Second arguement will be the image which will be pasted (not blended) on the background\n
       The resultPath arg is where the user wants the image to output to'''
    def __init__(self, bgImgPath : Path, fgImgPath : Path, resultPath : Path): # Class constructor
        # Are the parameters directories or single files?
        if Path.is_dir(bgImgPath):
            self.bgImgPaths = list(bgImgPath.iterdir())
        else:
            self.bgImgPaths = [bgImgPath]
        if Path.is_dir(fgImgPath):
            self.fgImgPaths = list(fgImgPath.iterdir())
        else:
            self.fgImgPaths = [fgImgPath]
        #################################################
        self.resultPath = resultPath
        self.imageId = 0
        self.annotationId = 0 
        self.info = { 
            "description": "This dataset was made by MakeWise employee",
            "url": "https://makewise.pt/",
            "version": "1.0",
            "year" : 2025,
            "contributor": "MakeWise",
            "data_created": "01-02-2025"
        }

        self.license = {
            "url": "https://www.gnu.org/licenses/gpl-3.0.html",
            "id": 1,
            "name": "GNU General Public License"
        }
        self.images = []
        self.annotations = []

        self.category = {
            "supercategory": "smoke",
            "id": 1,
            "name": "smoke"
        }
        if not Path.exists(resultPath): # if the result path doesn't exist, the class constructor will create one
            Path.mkdir(resultPath) 




    # Iterating thorugh the id. This function is mainly used for the json
    def __nextImageId(self):
        self.imageId += 1 
        return self.imageId
    
    def __nextAnnotationId(self):
        self.annotationId += 1
        return self.annotationId
    ######################################################################

    # Apply the filters on the expected Image argument and returned it along with its random coordinates to later be pasted on the other image
    def randomizeImage(self, smokeImg, forestImg):
        scale = random.uniform(0.05, 0.3) # Diminuir a imagem aleatoriamente, ou seja, com um scale aleatorio
        smokeImg_width = int(forestImg.shape[1] * scale)
        smokeImg_height = int(forestImg.shape[0] * scale)
        smokeImg = cv2.resize(smokeImg, (smokeImg_height, smokeImg_width))

        # Flip
        if (random.randrange(0, 4) == 2): # 33%
            smokeImg = cv2.flip(smokeImg, 1) # Girar a imagem verticalmente, baseado numa probabilidade.

        # Rotation with Matrix2D
        matrix = cv2.getRotationMatrix2D(((smokeImg.shape[1]-1)/2.0,(smokeImg.shape[0]-1)/2.0),random.randrange(0, 10),1)
        smokeImg = cv2.warpAffine(smokeImg,matrix,(smokeImg.shape[1],smokeImg.shape[0]))

        # Random Location
        r_x = random.randrange(0, forestImg.shape[1] - smokeImg.shape[1])
        r_y = random.randrange(min(max(forestImg.shape[0]//2- smokeImg.shape[0],0),forestImg.shape[0]- smokeImg.shape[0])-1, forestImg.shape[0]- smokeImg.shape[0])

        return (r_x, r_y, smokeImg)
    


    # Create a segmentation on the given image and return its coordinates
    def segmentate(self, readImg : Image):
        red, green, blue, alpha = readImg.split()
        rootImg = np.array(alpha)

        _,threshold = cv2.threshold(rootImg, 0, 255, cv2.THRESH_BINARY) # Threshold based on the alpha channel

        # SEGMENTATION
        contours, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Encontrar os contours -> (thres.)
        cnt = max(contours, key = cv2.contourArea)
        approx = cv2.approxPolyDP(cnt, 0.004 * cv2.arcLength(cnt, True), True)

        return approx
        

    # Create and return the coords of the given image
    def getBbox(self, readImg : Image):
        red, green, blue, alpha = readImg.split()
        rootImg = np.array(alpha)

        _,threshold = cv2.threshold(rootImg, 0, 255, cv2.THRESH_BINARY) # Threshold based on the alpha channel
        contours, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Encontrar os contours -> (thres.)

        x,y,w,h = cv2.boundingRect(rootImg)

        return np.array([[x,y],[w,h]]) # Return a matrix of the coordinates
    

    def writeJSON(self, jsonPath : Path):
        '''Writes a JSON in a COCO JSON format.\n
           The first argument is where the JSON is going to be written to.'''
        jsonPath = Path(jsonPath) # if the json is not a Path like object
        if (not jsonPath.exists()):
            with open(jsonPath, "w") as json_file:
                file = {
                    "info": self.info,
                    "license": [self.license],
                    "images": self.images,
                    "annotations": self.annotations,
                    "categories": [self.category]
                }
                json.dump(file, json_file)
        else:
            print("The Json already exists")

    def mergeJSONs(self, insertTo_JSON : Path, extractFrom_JSON : Path):
        '''Merges two JSONs, the first arguement is the JSON where the user wants to insert the data,\n 
        the second arguement is the JSON where the user wants to extract the data from so it can be pasted on the first refered JSON.'''
        with open(extractFrom_JSON, "r") as json_file:
            extractedContent = json.load(json_file) # We want the json array

            # Annotations merge vars
            ann_mergingArray = []
            ann_coordsArray = []
            ann_json_object = {}

            # Image merge vars
            mergingArray = []
            json_object = {}
            for obj in extractedContent: # Store every object inside the eextracted JSON inside an array
                json_object["license"] = 1
                json_object["id"] = self.__nextImageId()
                json_object["width"] = obj["image_width"]
                json_object["height"] = obj["image_height"]
                json_object["filename"] = obj["file_name"]
                json_object["category_id"] = 1
                json_object["coco_url"] = "http://random_url.com"
                json_object["data_captured"] = "2013-11-14 17:02:52"
                json_object["flickr_url"] = "http://random_flickr_url.com"

                # Annotations
                if (obj["bounding_area"] is not None): # Ignore the object if it doesn't have any bouding area
                    for coords in obj["bounding_area"]: # Store the coords of the bounding area
                        x = coords["x"]
                        y = coords["y"]
                        ann_coordsArray.append(x)
                        ann_coordsArray.append(y)
                    ann_json_object["id"] = self.__nextAnnotationId()
                    ann_json_object["image_id"] = self.imageId

                    
                                


                    ann_json_object["bbox"] = [123,123,123,123] # Hardcoded, for now.
                    ann_json_object["area"] = 123 # Hardcoded, for now. 
                    ann_json_object["segmentation"] = [ann_coordsArray]

                    ann_mergingArray.append(ann_json_object) # Annotation appending
                    ann_coordsArray = []
                    ann_json_object = {}


                mergingArray = np.append(mergingArray, json_object) # Image appending
                json_object = {}


        with open(insertTo_JSON, "r") as json_file: # Read and store the content of the JSON where the user wants to insert the data
            insertedJson = json.load(json_file)
            for obj in mergingArray:
                insertedJson["images"].append(obj) 
            for obj in ann_mergingArray:
                insertedJson["annotations"].append(obj)
        with open(insertTo_JSON, "w") as json_file: # Write the merged JSON
            json.dump(insertedJson, json_file)
        print("Merged ", insertTo_JSON.name, " with ", extractFrom_JSON.name)







    
    #This functions blends the fore ground with the background. The blending "config", how the blending occurs that is,
    #is defined by the functions above. For example, the randozimeImage, randozimes where the foreground will be, its scale '''
    def blendAndAnnotate(self):
        '''Paste the given foreground and background images together and stores the annotations in a given array which
        will be later written to a JSON file when the user calls the writeJSON method.
        The pasting is randomized on scale, position, rotation, etc. 
        All this happens in the randomizeImage method.'''
        fgChoice = random.choice(self.fgImgPaths)
        bgChoice = random.choice(self.bgImgPaths)

        
        # Read the images with cv2
        fgImg = cv2.imread(fgChoice, cv2.IMREAD_UNCHANGED) # Carrego a imagem buscada aleatoriamente
        bgImg = cv2.imread(bgChoice)
        bgImg = cv2.cvtColor(bgImg, cv2.COLOR_BGR2RGB)


        ############################################################ FILTERS
        rX, rY, fgImg = self.randomizeImage(fgImg, bgImg)
        ############################################################ FILTERS

        # Get offsets
        x_offset = rX
        y_offset = rY

        # Load the arrays to a Pillow Image Object
        foreground = Image.fromarray(fgImg,mode="RGBA")
        background = Image.fromarray(bgImg,mode="RGB")
        segCoords = self.segmentate(foreground)

        # Create the segmentation, and bounding boxes
        segCoords = segCoords + (x_offset, y_offset) # Get the segmentation coords
        box = self.getBbox(foreground)
        box = box + (x_offset, y_offset) # Get the bounding box coords

        # Key declaring and value equality for the JSON
        
 
        image_id = self.__nextImageId()
        annotation_id = self.__nextAnnotationId()
        height = bgImg.shape[0]
        width = bgImg.shape[1]
        segmentation = segCoords.flatten().tolist()
        bbox = np.array([box[0],[abs(box[1][0] - box[0][0]),abs(box[1][1] - box[0][1])]]).flatten().tolist()
        bbox_area = abs(box[1][0] - box[0][0]) * abs(box[1][1] - box[0][1])
        category_id = 1 # Its always smoke

        background.paste(foreground, (x_offset, y_offset),foreground) # Blend the smoke and forest images
        
        filename = "%05d_%s_%s.jpg"%(image_id, bgChoice.name.split(".")[0],fgChoice.name.split(".")[0])
        print(filename)
        background.save(self.resultPath.joinpath(filename))


        # Create the objects to be added to the JSON
        img_object = {
                "license": 1,
                "id": image_id,
                "width": width,
                "height": height,
                "filename": filename,   
                "category_id": category_id,
                "coco_url": "http://random_url.com",
                "height": height,
                "data_captured": "2013-11-14 17:02:52",
                "flickr_url": "http://random_flickr_url.com"
            }

        annotation_object = {
                "id": annotation_id,
                "image_id": image_id,
                "area": int(bbox_area),
                "bbox": bbox,
                "segmentation": [segmentation],
                "iscrowd": 0,
                "category_id": category_id
            }

        self.images.append(img_object)
        self.annotations.append(annotation_object)
    

    
        
    

    
    