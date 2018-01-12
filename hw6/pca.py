import numpy as np
from skimage import io
import sys

def do_the_avg_image(image_path = 'Aberdeen/*.jpg'):
    all_img_name = io.ImageCollection(image_path)
    im_size = all_img_name[0].shape
    all_data = []
    for image in all_img_name:
        all_data.append(image.flatten())
    img_avg =  np.mean(all_data, axis=0)
#     img_avg = np.mean(all_data, axis=0).astype(np.uint8)
#     img_avg = img_avg.reshape(im_size)
    #io.imshow(img_avg)
#     io.show()
#     io.imsave('average.jpg',img_avg)
    return all_img_name,all_data,img_avg

def Dimension_Reduction(all_data,img_avg):
    M = all_data - img_avg
    U, s, V = np.linalg.svd(M.T, full_matrices=False)
    weights = np.dot(M, U)
    return U, s, V,weights

def do_best_item_Eigenface(item,U,size):
    for i in range(item):
        eigenFace = np.copy(U[:, i].reshape(size))
        eigenFace *= -1
        eigenFace -= np.min(eigenFace)
        eigenFace /= np.max(eigenFace)
        eigenFace = (eigenFace * 255).astype(np.uint8)
#         io.imshow(eigenFace)
#         io.show()
#         io.imsave('eigenface_' + str(i) + '.jpg', eigenFace)

def do_Reconstruct(index,img_avg,U,weights,size):    
    reconstruction  = img_avg + np.dot(weights[index, :4], U[:, :4].T)
    reconstruction  = reconstruction.reshape(size)
    reconstruction  -= np.min(reconstruction )
    reconstruction  /= np.max(reconstruction )
    reconstruction  = (reconstruction  * 255).astype(np.uint8)
#     io.show()
#     io.imshow(reconstruction)
    io.imsave('reconstruction.jpg',reconstruction)


if __name__ == '__main__':
    image_path = sys.argv[1]
    #print(image_path)
    image_path = sys.argv[1]+"/*.jpg"
    all_img_name,all_data,img_avg = do_the_avg_image(image_path)
    size = all_img_name[0].shape
    U, s, V,weights = Dimension_Reduction(all_data,img_avg)
    item_path = sys.argv[2]
    #print(type(item_path))
    item = int(item_path[:-4])
    do_Reconstruct(item,img_avg,U,weights,size)