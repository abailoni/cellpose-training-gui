from traincellpose.io.images import read_uint8_img, write_image_to_file
import os

from traincellpose.preprocessing import normalize_image

if __name__ == '__main__':


    # /scratch/bailoni/projects/nastia/2022-03-07/train_cellpose/ann_project_recovered/Cellpose
    for root, dirs, files in os.walk("/Users/alberto-mac/Documents/DA_ESPORTARE/LOCAL_EMBL_FILES/scratch/projects/nastia/2022-03-07/train_cellpose/ann_project_recovered/Cellpose/training_images", topdown=False):
        for name in files:
            if name.endswith(".tif"):
                image_path = os.path.join(root, name)
                image = read_uint8_img(image_path)
                print(image.shape)
                # image[..., 2] = 0
                image[..., 1] = normalize_image(image[..., 1])
                out_image_path = os.path.join(root, "../new_images", name)

                os.makedirs(os.path.split(out_image_path)[0], exist_ok=True)
                write_image_to_file(out_image_path, image)

                # labels_path = os.path.join(root, "../training_labels", name.replace(".tif", "_masks.tif"))
                # labels = read_uint8_img(labels_path)




