from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import arabic_reshaper
from bidi.algorithm import get_display
# Open the desired Image you want to add text on
image = Image.open('White_image.jpg')

# open and read txt file 
my_file = open("sentences.txt", "r",encoding="utf8")
data = my_file.read()

#split data into list
data_into_list = data.replace('\n', ' ').split(" ")
data_into_list = list(data_into_list)
for word in data_into_list:
    #image = Image.open('White_image.jpg')
# # Call draw Method to add 2D graphics in an image
    #text_to_be_reshaped = 'مثلا روم زوم کنی بوم بوم کنه قلبم'
    counter = 0
    text_to_be_reshaped = word
    #print(word)

    #reshaped_text = arabic_reshaper.reshape(text_to_be_reshaped) 
    reshaped_text = arabic_reshaper.reshape(text_to_be_reshaped)

    bidi_text = get_display(reshaped_text)
    #Im = ImageDraw.Draw(image)
    image_font = ImageFont.truetype('ketab.ttf', 150) # second parameter increase the size of text
    #image = Image.new('RGBA', (800, 600), (255,255,255,0))
    image_draw = ImageDraw.Draw(image)
    image_draw.text((10,10), bidi_text, fill=(0,0,0), font=image_font)
    file_name = str(counter) + "image.png" 
    image.save(file_name)
    counter = counter+1 
    #image.close()
    # Add Text to an image
#Im.text((180, 270), reshaped_text,fill=(0, 0, 0),font=image_font)

# # Display edited image
image.show()

# # Save the edited image
