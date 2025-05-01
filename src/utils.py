def crop_to_multiple_of_16(img):
    width, height = img.size
    
    # Calculate new dimensions that are multiples of 8
    new_width = width - (width % 16)  
    new_height = height - (height % 16)
    
    # Calculate crop box coordinates
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = left + new_width
    bottom = top + new_height
    
    # Crop the image
    cropped_img = img.crop((left, top, right, bottom))
    
    return cropped_img
