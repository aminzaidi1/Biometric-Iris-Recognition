import cv2
import os
import time
import sys
from utils import preprocess, find_best_match, select_image_from_dataset, display_images, show_details

# Main function
if __name__ == '__main__':
    dataset_folder = r'..\Iris-Recognition\dataset'
    
    print("Please select an image from the dataset for matching:")
    query_image, individual_id, session_id = select_image_from_dataset(dataset_folder)
    
    if query_image is not None:
        start_time = time.time()  # Start measuring time
        
        best_match_id, similarity = find_best_match(dataset_folder, query_image)
        
        end_time = time.time()  # Stop measuring time
        execution_time = end_time - start_time

        # Load and display the input image
        query_images = [query_image]
        print("Displaying input image:")
        display_images(query_images)
        
        print("Best match found:")
        print(f"Individual ID: {best_match_id}")
        print(f"Similarity: {similarity}")
        print(f"Execution Time: {execution_time} seconds")
        
        
        # Display segmentation and normalization process details for the input image
        print("Displaying segmentation and normalization process details for the input image:")
        show_details(query_image)

        if best_match_id is not None:
            print(f"Displaying all images of the matched individual {best_match_id}:")
            individual_images = []
            for filename in os.listdir(dataset_folder):
                if filename.startswith(best_match_id) and filename.endswith(".jpg"):
                    filepath = os.path.join(dataset_folder, filename)
                    img = cv2.imread(filepath)
                    individual_images.append(img)
            display_images(individual_images)

        # Close all OpenCV windows after processing
        cv2.destroyAllWindows()