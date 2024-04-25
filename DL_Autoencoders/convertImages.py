import numpy as np

def split_images(images):
    """ Split each image in the batch into its RGB components,
    each becoming a single-channel image with its
    corresponding label intact """
    red_channel = images[:, :, :, 0:1]
    green_channel = images[:, :, :, 1:2]
    blue_channel = images[:, :, :, 2:3]

    # concatenate along the batch axis
    new_images = np.concatenate((red_channel, green_channel, blue_channel), axis=0)
    return new_images

def split_labels(labels):
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)
    labels = labels.astype(str)
    labels = np.char.zfill(labels, 3)

    labels = np.array([list(map(int, list(row[0]))) for row in labels])
    redLabels = [x[0] for x in labels]
    greenLabels = [x[1] for x in labels]
    blueLabels = [x[2] for x in labels]
    new_labels = np.concatenate([labels[0], labels[1], labels[-1]])
    new_labels = np.concatenate([redLabels, greenLabels, blueLabels])
    return new_labels

def merge_images(single_channel_images):
    """Merge each set of three single-channel images
    back into RGB images and concat labels
    back to the original form."""
    batch_size = single_channel_images.shape[0] // 3
    
    # Reshape the images back into RGB
    merged_images = np.stack((
        single_channel_images[:batch_size, :, :, 0], 
        single_channel_images[batch_size:2*batch_size, :, :, 0], 
        single_channel_images[2*batch_size:, :, :, 0]
    ), axis=-1)
       
    return merged_images

def merge_labels(labels):
    batch_size = labels.shape[0] // 3
    # Reshape the labels back into RGB
    original_labels = np.stack((
        labels[:batch_size], 
        labels[batch_size:2*batch_size], 
        labels[2*batch_size:]
    ), axis=-1)
    return original_labels