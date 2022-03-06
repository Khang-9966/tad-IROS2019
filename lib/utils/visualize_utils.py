import matplotlib.pyplot as plt
# import cv2
from PIL.ImageDraw import Draw
import PIL
import copy
import numpy as np

def draw_rectangle(draw, x1y1x2y2, color, width=1):
    for i in range(width):
        rect_start = [x1y1x2y2[0][0] - i, x1y1x2y2[0][1] - i]
        rect_end = [x1y1x2y2[1][0] + i, x1y1x2y2[1][1] + i]
#         print(rect_start)
#         print(rect_end)
#         print(tuple(rect_start+rect_end))
        draw.rectangle(rect_start+rect_end, outline = tuple(color))
        
def vis_multi_prediction(image, 
                         pred_boxes_t, 
                         all_observed_boxes,
                         all_observed_track_id,
                         save_dir,
                         track_ids_to_show=None,
                         plot_gt=True,
                         width=2,
                         show_id=True
                         ):
    '''
    Params:
        pred_boxes_t: a dictionary 
        all_observed_boxes: a dictionary
        tracker_colors: a numpy array (num_cars, 3)
    '''
    H, W, n_channels = np.asarray(image).shape
    # try:
    #     print(all_observed_boxes)
    #     all_observed_track_id = list(all_observed_boxes.keys())
    # except:
    #     # if no boxes observed, bbox anomaly score is 0
    #     return 0
    
    all_observed_boxes_temp = {}
    for box_ in all_observed_boxes:
      all_observed_boxes_temp[box_[0]] = box_[1:]
    tracker_colors = np.random.randint(255, size=(len(np.unique(all_observed_track_id)), 4))
  

    image = PIL.Image.fromarray(image[:,:,::-1])
    draw = Draw(image)
    if plot_gt:
        for track_id in all_observed_track_id: 
          try:
            target_bbox = all_observed_boxes_temp[track_id].copy() # (1, 4) torch cuda tensor, convert to numpy
            target_bbox = np.expand_dims(target_bbox,0)
            target_bbox[:,0] = target_bbox[:,0] * W
            target_bbox[:,1] = target_bbox[:,1] * H
            target_bbox[:,2] = target_bbox[:,2] * W
            target_bbox[:,3] = target_bbox[:,3] * H
            # target_bbox = target_bbox.detach().cpu().numpy()
            target_bbox = target_bbox[0]
            xmin = int(target_bbox[0] - target_bbox[2]/2)
            ymin = int(target_bbox[1] - target_bbox[3]/2)
            xmax = int(target_bbox[0] + target_bbox[2]/2)
            ymax = int(target_bbox[1] + target_bbox[3]/2)


            draw_rectangle(draw, [(xmin,ymin),(xmax,ymax)], (255,0,0), width=width)    
          except:
            continue   
    
    for track_id in pred_boxes_t.keys():
#         print(track_id)
        
        if track_ids_to_show is not None and track_id not in track_ids_to_show:
            continue
#         print("Drawing...")
        predictions = pred_boxes_t[track_id].copy() # (n, 4) troch cuda tensor, convert to numpy
       
        if len(predictions) == 0:
            continue
      
        predictions = np.array(predictions)
        predictions[:,0] = predictions[:,0] * W
        predictions[:,1] = predictions[:,1] * H
        predictions[:,2] = predictions[:,2] * W
        predictions[:,3] = predictions[:,3] * H
        print(predictions)
        for i, pred_box in enumerate(predictions):
            xmin = int(pred_box[0] - pred_box[2]/2)
            ymin = int(pred_box[1] - pred_box[3]/2)
            xmax = int(pred_box[0] + pred_box[2]/2)
            ymax = int(pred_box[1] + pred_box[3]/2)
            print([(xmin,ymin),(xmax,ymax)])
            draw_rectangle(draw, [(xmin,ymin),(xmax,ymax)], (0,255,0), width=width)
        
        if show_id:
            draw.text((xmin, ymin),str(track_id),(0,0,255))
      
  
    image.save(save_dir)
    #draw.save(save_dir)   