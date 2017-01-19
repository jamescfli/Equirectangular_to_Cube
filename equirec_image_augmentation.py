# import cv2
from PIL import Image as pil_image
from loop_destination_map_to_in_image import convert_back_cv2_wrapper, cut_body_part, cut_top_face
import numpy as np
import scipy.ndimage as ndi


def random_vshift(img_src_in_array, vshift_range_limit,
                  row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.):
    """Performs a random shift in vertical direction of a Numpy image.
    # Arguments
        img_src_in_array: Input tensor. Must be 3D.
        vshift_range_limit: shift range, as a float fraction of the height
        row_axis: Index of axis for rows in the input array.
        col_axis: Index of axis for columns in the input array.
        channel_axis: Index of axis for channels in the input array.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        Shifted Numpy image tensor.
    """
    # note img array axis sequence should be (h, w, ch) as in OpenCV
    h, w = img_src_in_array.shape[0:2]
    tx = np.random.uniform(-vshift_range_limit, +vshift_range_limit) * h
    # print tx    # debug
    translation_matrix = np.array([[1, 0, tx],
                                   [0, 1, 0],
                                   [0, 0, 1]])
    img_vshifted_in_array = apply_transform(img_src_in_array, translation_matrix, channel_axis, fill_mode, cval)
    return img_vshifted_in_array


def random_rotation(img_src_in_array, rg, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.):
    """Performs a random rotation of a Numpy image.
    # Arguments
        img_src_in_array: Input tensor. Must be 3D.
        rg: Rotation range, in degrees (<= 180)
        row_axis: Index of axis for rows in the input.
        col_axis: Index of axis for columns in the input.
        channel_axis: Index of axis for channels in the input.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        Rotated Numpy image.
    """
    theta = np.pi / 180 * np.random.uniform(-rg, rg)
    # print theta/np.pi*180   # debug
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])

    h, w = img_src_in_array.shape[row_axis], img_src_in_array.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
    img_src_in_array = apply_transform(img_src_in_array, transform_matrix, channel_axis, fill_mode, cval)
    return img_src_in_array


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def apply_transform(x, transform_matrix, channel_axis=0, fill_mode='nearest', cval=0.):
    x = np.rollaxis(x, channel_axis, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(x_channel, final_affine_matrix,
                                                         final_offset, order=0, mode=fill_mode, cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x


if __name__ == '__main__':
    # img_src = cv2.imread('Equi_Images/livingroom_1024x512.jpg', cv2.IMREAD_COLOR)
    # img_top_face = cut_top_face(convert_back_cv2_wrapper(img_src))
    # img_body_part = cut_body_part(img_src)
    # cv2.imwrite('Output_Images/livingroom_top.jpg', img_top_face)
    # cv2.imwrite('Output_Images/livingroom_body.jpg', img_body_part)

    # # v-shift with zero padding
    # img_body_src = pil_image.open('Output_Images/livingroom_body.jpg')
    # shift_range = 0.1
    # img_vshifted = random_vshift(img_src_in_array = np.array(img_body_src), vshift_range_limit = shift_range)
    # pil_image.fromarray(img_vshifted.astype('uint8')).save('Output_Images/body_vshifted_hrl{}.jpg'.format(shift_range))

    # # rotation to the top face
    # img_top_src = pil_image.open('Output_Images/livingroom_top.jpg')
    # rotation_limit_in_degree = 180  # clockwise is positive rotation
    # img_rotated = random_rotation(img_src_in_array=np.array(img_top_src), rg=rotation_limit_in_degree)
    # pil_image.fromarray(img_rotated.astype('uint8')).save('Output_Images/top_rotated_rgl{}.jpg'
    #                                                       .format(rotation_limit_in_degree))

    # concatenate body
    img_vshifted_in_array = np.array(pil_image.open('Output_Images/body_vshifted_hrl0.1.jpg'), dtype='uint8')
    img_rotated_in_array = np.array(pil_image.open('Output_Images/top_rotated_rgl180.jpg'), dtype='uint8')
    img_concat_in_array = np.hstack((img_vshifted_in_array, img_rotated_in_array))
    pil_image.fromarray(img_concat_in_array).save('Output_Images/aug_body_top_concat.jpg')