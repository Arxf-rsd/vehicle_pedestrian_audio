import openvino as ov
import cv2
import numpy as np
import matplotlib.pyplot as plt


core = ov.Core()

model_pv = core.read_model(model='models/pedestrian-and-vehicle-detector-adas-0001.xml')
compiled_model_pv = core.compile_model(model = model_pv, device_name="CPU")

input_layer_pv = compiled_model_pv.input(0)
output_layer_pv = compiled_model_pv.output(0)

print("Input shape:", input_layer_pv.shape)
print("Output shape:", output_layer_pv.shape)

model_ag = core.read_model(model='models/age-gender-recognition-retail-0013.xml')
compiled_model_ag = core.compile_model(model = model_ag, device_name="CPU")

input_layer_ag = compiled_model_ag.input(0)
output_layer_ag = compiled_model_ag.output

print("Input shape:", input_layer_ag.shape)
print("Output shape:", output_layer_ag)


def preprocess(image, input_layer_pv):
    N, input_channels, input_height, input_width = input_layer_pv.shape
    resized_image = cv2.resize(image, (input_width, input_height))
    transposed_image= resized_image.transpose(2, 0, 1)
    input_image = np.expand_dims(transposed_image,0)
    return input_image

def find_pvboxes(image, results, confidence_threshold):
     results = results.squeeze()
     scores = results[:,2]
     boxes = results[:,-4:]
     pv_boxes = boxes[scores >= confidence_threshold]
     scores = scores[scores >= confidence_threshold]
     image_h, image_w, image_channels = image.shape
     pv_boxes = pv_boxes*np.array([image_w, image_h, image_w, image_h])
     pv_boxes = pv_boxes.astype(np.int64)
     return pv_boxes, scores

def draw_age_gender(pv_boxes, image):

    show_image = image.copy()
    
    for i in range(len(pv_boxes)):
        xmin, ymin, xmax, ymax = pv_boxes[i]
        pv = image[ymin:ymax, xmin:xmax]

        #--- age and gender ---
        input_image_ag = preprocess(image, input_layer_ag)
        results_ag = compiled_model_ag([input_image_ag])
        age, gender = results_ag[1], results_ag[0]
        age = np.squeeze(age)
        age = int(age*100)

        gender = np.squeeze(gender)

        if(gender[0]>=0.65):
            gender = "pedestrian"
            box_color = (200, 200, 0)
            
        elif(gender[1]>=0.55):
            gender = "vehicle"
            box_color = (0, 200, 200)
        else:
            gender = "unkown"
            box_color = (200, 200, 200)
        
       #--- age and gender ---
        fontScale = image.shape[1]/750

        text = gender +''+ str(age)
        cv2.putText(show_image, text, (xmin, ymin),cv2.FONT_HERSHEY_SIMPLEX, 7, (0, 200, 0), 10)
        cv2.rectangle(img=show_image, pt1=(xmin, ymin), pt2=(xmax, ymax), color=box_color, thickness=10)
    return show_image


def predict_image(image, conf_threshold):
    input_image = preprocess(image, input_layer_pv)
    results = compiled_model_pv([input_image])[output_layer_pv]
    pv_boxes, scores = find_pvboxes(image, results, conf_threshold)
    visualize_image = draw_age_gender(pv_boxes, image)
    return visualize_image
