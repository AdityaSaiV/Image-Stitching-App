import anvil.server
import keras
from tensorflow.keras.utils import load_img
import anvil.media
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import io


anvil.server.connect("server_3UAL3FGB2TQVS6ILQVCKL4N5-4VSIRZSYHB3T5CXQ")

@anvil.server.callable
def image_stitch(file):
    with anvil.media.TempFile(file) as filename:
        img = load_img(filename)
        img.save('Query.jpg')
        
@anvil.server.callable
def image_stitch2(file):
    with anvil.media.TempFile(file) as filename:
        img2 = load_img(filename)
        img2.save('Train.jpg')

@anvil.server.callable
def stitching_function():
    train_photo = cv2.imread('./' + 'Train.jpg')
    train_photo = cv2.cvtColor(train_photo, cv2.COLOR_BGR2RGB)
    train_photo_gray = cv2.cvtColor(train_photo, cv2.COLOR_RGB2GRAY)
    query_photo = cv2.imread('./' + 'Query.jpg')
    query_photo = cv2.cvtColor(query_photo, cv2.COLOR_BGR2RGB)
    query_photo_gray = cv2.cvtColor(query_photo, cv2.COLOR_RGB2GRAY)

    def select_descriptor_methods(image, method=None):    
        if method == 'sift':
            descriptor = cv2.SIFT_create()
        elif method == 'surf':
            descriptor = cv2.SURF_create()
        elif method == 'brisk':
            descriptor = cv2.BRISK_create()
        elif method == 'orb':
            descriptor = cv2.ORB_create()

        (keypoints, features) = descriptor.detectAndCompute(image, None)
        return (keypoints, features)

    keypoints_train_img, features_train_img = select_descriptor_methods(train_photo_gray, method='sift')
    keypoints_query_img, features_query_img = select_descriptor_methods(query_photo_gray, method='sift')

    for keypoint in keypoints_query_img:
        x,y = keypoint.pt
        size = keypoint.size 
        orientation = keypoint.angle
        response = keypoint.response 
        octave = keypoint.octave
        class_id = keypoint.class_id

    keypoint_train = cv2.drawKeypoints(train_photo_gray, keypoints_train_img, None, color=(0, 255, 0))
    keypoint_query = cv2.drawKeypoints(query_photo_gray, keypoints_query_img, None, color=(0, 255, 0))
    
    keypoint_train = cv2.cvtColor( keypoint_train, cv2.COLOR_BGR2RGB)
    keypoint_query = cv2.cvtColor(keypoint_query, cv2.COLOR_BGR2RGB)
    
    cv2.imwrite('keypoint_train.jpg',keypoint_train)
    cv2.imwrite('keypoint_query.jpg',keypoint_query)

    def create_matching_object(method,crossCheck):
        if method == 'sift' or method == 'surf':
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=crossCheck)
        elif method == 'orb' or method == 'brisk':
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=crossCheck)
        return bf

    def key_points_matching(features_train_img, features_query_img, method):
        bf = create_matching_object(method, crossCheck=True)
        best_matches = bf.match(features_train_img,features_query_img)
        rawMatches = sorted(best_matches, key = lambda x:x.distance)
        #print("Raw matches with Brute force):", len(rawMatches))
        return rawMatches

    def key_points_matching_KNN(features_train_img, features_query_img, ratio, method):
        bf = create_matching_object(method, crossCheck=False)
        rawMatches = bf.knnMatch(features_train_img, features_query_img, k=2)
        #print("Raw matches (knn):", len(rawMatches))
        matches = []
        for m,n in rawMatches:
            if m.distance < n.distance * ratio:
                matches.append(m)
        return matches

    feature_to_match = 'knn'
    feature_extraction_algo = 'sift'
    #print("Drawing: {} matched features Lines".format(feature_to_match))
    if feature_to_match == 'bf':
        matches = key_points_matching(features_train_img, features_query_img, method=feature_extraction_algo) 
        mapped_features_image = cv2.drawMatches(train_photo,keypoints_train_img,query_photo,keypoints_query_img,matches[:100],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    elif feature_to_match == 'knn':
        matches = key_points_matching_KNN(features_train_img, features_query_img, ratio=0.75, method=feature_extraction_algo)
        mapped_features_image_knn = cv2.drawMatches(train_photo, keypoints_train_img, query_photo, keypoints_query_img, np.random.choice(matches,100),None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    mapped = cv2.cvtColor( mapped_features_image_knn, cv2.COLOR_BGR2RGB)
    cv2.imwrite('knn_match.jpg',mapped)
    
    
    def homography_stitching(keypoints_train_img, keypoints_query_img, matches, reprojThresh):   

        keypoints_train_img = np.float32([keypoint.pt for keypoint in keypoints_train_img])
        keypoints_query_img = np.float32([keypoint.pt for keypoint in keypoints_query_img])
        if len(matches) > 4:
            points_train = np.float32([keypoints_train_img[m.queryIdx] for m in matches])
            points_query = np.float32([keypoints_query_img[m.trainIdx] for m in matches])
            (H, status) = cv2.findHomography(points_train, points_query, cv2.RANSAC, reprojThresh)

            return (matches, H, status)
        else:
            return None

    M = homography_stitching(keypoints_train_img, keypoints_query_img, matches, reprojThresh=4)

    if M is None:
        print("Error!")

    (matches, Homography_Matrix, status) = M


    width = query_photo.shape[1] + train_photo.shape[1]
    height = max(query_photo.shape[0], train_photo.shape[0])
    result = cv2.warpPerspective(train_photo, Homography_Matrix,  (width, height))
    result[0:query_photo.shape[0], 0:query_photo.shape[1]] = query_photo
    
    result = cv2.cvtColor( result, cv2.COLOR_BGR2RGB)
    cv2.imwrite('final.jpg',result)

@anvil.server.callable
def keypoint_image1():
    image = Image.open('keypoint_train.jpg')
    bs=io.BytesIO()
    name = 'image_result'
    image.save(bs,format='png')
    return anvil.BlobMedia('image/png',bs.getvalue(),name=name)
@anvil.server.callable
def keypoint_image2():
    image = Image.open('keypoint_query.jpg')
    bs=io.BytesIO()
    name = 'image_result'
    image.save(bs,format='png')
    return anvil.BlobMedia('image/png',bs.getvalue(),name=name)
@anvil.server.callable
def knn_match():
    image = Image.open('knn_match.jpg')
    bs=io.BytesIO()
    name = 'image_result'
    image.save(bs,format='png')
    return anvil.BlobMedia('image/png',bs.getvalue(),name=name)
@anvil.server.callable
def final():
    image = Image.open('final.jpg')
    bs=io.BytesIO()
    name = 'image_result'
    image.save(bs,format='png')
    return anvil.BlobMedia('image/png',bs.getvalue(),name=name)
