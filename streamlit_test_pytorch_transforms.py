import streamlit as st
from PIL import Image
from torchvision import transforms
import torch

## Layout of the app
st.title('Check your PyTorch augmented images here before training them!')
st.write('Researchers working with PyTorch often perform image augmentations in their experiments.' 
         'Some augmentations do not make sense in the context of the experiment.'
         'Check your augmented images here before training your CNNs.' 
         'Refer to this link https://pytorch.org/docs/stable/torchvision/transforms.html for more info.')

image = Image.open('data/data_augmentation.png')
st.image(image, use_column_width=True)
st.write('Please input the original image into the sidebar.')
uploaded_image = st.file_uploader("Upload your image here", type="jpg")
option = st.selectbox('What image transform would you want to try out?',
                      ('CenterCrop', 'ColorJitter', 'Grayscale', 'Pad', 'RandomHorizontalFlip','RandomVerticalFlip','RandomAffine', 'RandomRotation', 'FiveCrop','TenCrop','RandomCrop', 'RandomResizedCrops'))
def is_grey_scale(img_path):
    img = Image.open(img_path).convert('RGB')
    w,h = img.size
    for i in range(w):
        for j in range(h):
            r,g,b = img.getpixel((i,j))
            if r != g != b:
                return False
    return True
def each_click_action(uploaded_image, option):
    im = Image.open(uploaded_image)
    img_height, img_width = im.size

    if option == 'CenterCrop':
        sub_option_width = st.sidebar.number_input("Width (Just FYI, the image width is "+ str(img_width)+")", 1, img_width, 256)
        sub_option_height = st.sidebar.number_input("Height (Just FYI, the image height is "+ str(img_height) +")", 1, img_height, 256)
        data_transforms = transforms.Compose([
            transforms.CenterCrop((sub_option_height, sub_option_width)),
            transforms.ToTensor()
        ])

        img = data_transforms(im)
        img_display = transforms.ToPILImage(mode='RGB')(img)
        return img_display

    if option == 'ColorJitter':
        sub_option_brightness = st.sidebar.slider("Brightness", 0.0, 1.0, 0.0, 0.1)
        sub_option_contrast = st.sidebar.slider("Contrast", 0.0, 1.0, 0.0, 0.1)
        sub_option_saturation = st.sidebar.slider("Saturation", 0.0, 1.0, 0.0, 0.1)
        sub_option_hue = st.sidebar.slider("Hue", 0.0, 0.5, 0.0, 0.1)
        data_transforms = transforms.Compose([
            transforms.ColorJitter(sub_option_brightness, sub_option_contrast, sub_option_saturation, sub_option_hue),
            transforms.ToTensor()
        ])
        img = data_transforms(im)
        img_display = transforms.ToPILImage(mode='RGB')(img)
        return img_display


    # TODO: Fix this!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if option == 'FiveCrop':
        sub_option_width = st.sidebar.number_input("Width (Just FYI, the image width is " + str(img_width) + ")", 1,
                                                   img_width, 224)
        sub_option_height = st.sidebar.number_input("Height (Just FYI, the image height is " + str(img_height) + ")", 1,
                                                   img_height, 224)
        data_transforms = transforms.Compose([
            transforms.FiveCrop(size = (sub_option_width, sub_option_height)),
            transforms.Lambda(lambda crops: torch.stack([(transforms.ToTensor()(crop)) for crop in crops]))
        ])

        # Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops]))

        imgs = data_transforms(im)
        img_display=[]
        for img in imgs:
            img_display.append(transforms.ToPILImage(mode='RGB')(img))
        return img_display

    # TODO: Fix this!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if option == 'TenCrop':
        sub_option_width = st.sidebar.number_input("Width (Just FYI, the image width is " + str(img_width) + ")", 1,
                                                   img_width, 224)
        sub_option_height = st.sidebar.number_input("Height (Just FYI, the image height is " + str(img_height) + ")", 1,
                                                   img_height, 224)
        data_transforms = transforms.Compose([
            transforms.TenCrop(size=(sub_option_width, sub_option_height)),
            transforms.Lambda(lambda crops: torch.stack([(transforms.ToTensor()(crop)) for crop in crops]))
        ])
        imgs = data_transforms(im)
        img_display = []
        for img in imgs:
            img_display.append(transforms.ToPILImage(mode='RGB')(img))
        return img_display

    if option == 'RandomCrop':
        sub_option_width = st.sidebar.number_input("Width (Just FYI, the image width is "+ str(img_width)+")", 1, img_width, 256)
        sub_option_height = st.sidebar.number_input("Height (Just FYI, the image height is "+ str(img_height) +")", 1, img_height, 256)

        data_transforms = transforms.Compose([
            transforms.RandomCrop((sub_option_height, sub_option_width)),
            transforms.ToTensor()
        ])

        img = data_transforms(im)
        img_display = transforms.ToPILImage(mode='RGB')(img)
        return img_display

    if option == 'RandomResizedCrops':
        sub_option_width = st.sidebar.number_input("Width (Just FYI, the image width is "+ str(img_width)+")", 1, img_width, 256)
        sub_option_height = st.sidebar.number_input("Height (Just FYI, the image height is "+ str(img_height) +")", 1, img_height, 256)
        # sub_option_scale = st.sidebar.slider("Scale", 0.01, 1.0, 0.08, 0.01)

        data_transforms = transforms.Compose([
            transforms.RandomResizedCrop(size=(sub_option_height,sub_option_width)),
            transforms.ToTensor()
        ])

        img = data_transforms(im)
        img_display = transforms.ToPILImage()(img)
        return img_display

    if option == 'RandomHorizontalFlip':
        sub_option_probability = st.sidebar.slider("Probability of flip", 0.0, 1.0, 0.5, 0.1)


        data_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=sub_option_probability),
            transforms.ToTensor()
        ])

        img = data_transforms(im)
        img_display = transforms.ToPILImage()(img)
        return img_display

    if option == 'RandomVerticalFlip':
        sub_option_probability = st.sidebar.slider("Probability of flip", 0.0, 1.0, 0.5, 0.1)
        data_transforms = transforms.Compose([
            transforms.RandomVerticalFlip(p=sub_option_probability),
            transforms.ToTensor()
        ])

        img = data_transforms(im)
        img_display = transforms.ToPILImage()(img)
        return img_display

    if option == 'RandomRotation':

        sub_option_degrees = st.sidebar.slider("Degrees", 0, 360, 0, 10)

        data_transforms = transforms.Compose([
            transforms.RandomRotation(degrees=sub_option_degrees,
                                      resample=False,
                                      expand=False,
                                      center=None),
            transforms.ToTensor()
        ])

        img = data_transforms(im)
        img_display = transforms.ToPILImage()(img)
        return img_display

    if option =='Grayscale':
        sub_option_num_channels = st.selectbox('Num channels',(1, 3))
        data_transforms = transforms.Compose([
            transforms.Grayscale(num_output_channels=sub_option_num_channels),
            transforms.ToTensor()
        ])

        img = data_transforms(im)
        img_display = transforms.ToPILImage()(img)
        return img_display

    if option == 'Pad':
        sub_option_left_padding = st.sidebar.number_input("Left padding (Just FYI, the image height is "+ str(img_height)+")", 1, img_width, 50)
        sub_option_top_padding = st.sidebar.number_input("Top padding (Just FYI, the image width is "+ str(img_width)+")", 1, img_width, 50)
        sub_option_right_padding = st.sidebar.number_input("Right padding (Just FYI, the image height is "+ str(img_height)+")", 1, img_width, 50)
        sub_option_bottom_padding = st.sidebar.number_input("Bottom padding (Just FYI, the image width is "+ str(img_width)+")", 1, img_width, 50)
        sub_option_padding_mode =  st.selectbox('Padding mode',('constant', 'edge', 'reflect', 'symmetric'))
        if sub_option_padding_mode == 'constant':
            if is_grey_scale(uploaded_image):
                sub_option_fill = st.sidebar.slider("Fill", 0, 255, 0, 1)
                data_transforms = transforms.Compose([transforms.Pad(padding=(sub_option_left_padding, sub_option_top_padding, sub_option_right_padding,
                                            sub_option_bottom_padding), fill=sub_option_fill, padding_mode=sub_option_padding_mode),
                                     transforms.ToTensor()])
                img = data_transforms(im)
                img_display = transforms.ToPILImage(mode='RGB')(img)
                return img_display
            else:
                sub_option_fill_red = st.sidebar.slider("Red Fill", 0, 255, 0, 1)
                sub_option_fill_green = st.sidebar.slider("Green Fill", 0, 255, 0, 1)
                sub_option_fill_blue = st.sidebar.slider("Blue Fill", 0, 255, 0, 1)

                data_transforms = transforms.Compose([
                    transforms.Pad(padding=(sub_option_left_padding, sub_option_top_padding, sub_option_right_padding,
                                            sub_option_bottom_padding), fill=(sub_option_fill_red, sub_option_fill_green, sub_option_fill_blue),
                                   padding_mode=sub_option_padding_mode),
                    transforms.ToTensor()
                ])

                img = data_transforms(im)
                img_display = transforms.ToPILImage(mode='RGB')(img)
                return img_display
        else:
            data_transforms = transforms.Compose([
                transforms.Pad(padding=(
                sub_option_left_padding, sub_option_top_padding, sub_option_right_padding, sub_option_bottom_padding),
                               padding_mode=sub_option_padding_mode),
                transforms.ToTensor()
            ])

            img = data_transforms(im)
            img_display = transforms.ToPILImage(mode='RGB')(img)
            return img_display

    # TODO: Fix this!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if option == 'RandomAffine':
        sub_option_randomaffine_degrees =  st.sidebar.slider("Degrees", 0, 180, 0, 1)
        sub_option_randomaffine_translate_horizontal = st.sidebar.slider("Horizontal translate", 0.0, 1.0, 0.0, 0.1)
        sub_option_randomaffine_translate_vertical = st.sidebar.slider("Vertical translate", 0.0, 1.0, 0.0, 0.1)
        sub_option_randomaffine_scale_horizontal =  st.sidebar.slider("Horizontal scale", 0.1, 1.0, 1.0, 0.1)
        sub_option_randomaffine_scale_vertical = st.sidebar.slider("Vertical scale", 0.1, 1.0, 1.0, 0.1)

        sub_option_randomaffine_sheer_horizontal_lower = st.sidebar.slider("Horizontal sheer lower cutoff", 0.0, 1.0, 0.0, 0.1)
        sub_option_randomaffine_sheer_horizontal_upper = st.sidebar.slider("Horizontal sheer upper cutoff", 0.0, 1.0, 0.0, 0.1)
        sub_option_randomaffine_sheer_vertical_lower = st.sidebar.slider("Vertical sheer lower cutoff", 0.0, 1.0, 0.0, 0.1)
        sub_option_randomaffine_sheer_vertical_upper = st.sidebar.slider("Vertical sheer upper cutoff", 0.0, 1.0, 0.0, 0.1)

        data_transforms = transforms.Compose([
            transforms.RandomAffine(degrees= sub_option_randomaffine_degrees,
                                    translate = (sub_option_randomaffine_translate_horizontal,sub_option_randomaffine_translate_vertical),
                                    scale= (sub_option_randomaffine_scale_horizontal,sub_option_randomaffine_scale_vertical),
                                    shear = (sub_option_randomaffine_sheer_horizontal_lower, sub_option_randomaffine_sheer_horizontal_upper,sub_option_randomaffine_sheer_vertical_lower,sub_option_randomaffine_sheer_vertical_upper)
                                    ),
            transforms.ToTensor()
        ])

        img = data_transforms(im)
        img_display = transforms.ToPILImage(mode='RGB')(img)
        return img_display

img_display = each_click_action(uploaded_image, option)
if (st.button('Show me the transformed image!!')):
    st.image(img_display)

