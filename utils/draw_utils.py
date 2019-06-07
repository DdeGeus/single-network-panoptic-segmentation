from PIL import Image, ImageDraw, ImageFont

FONT = ImageFont.load_default()

def draw_labels_and_probs(draw_obj, box, label, prob, color):
  x, y = box[0], box[1]
  draw_obj.rectangle([x, y, x+100, y+10],
                     fill=color)

  txt = label + ": " + str(prob)
  draw_obj.text(xy=(x, y),
                text=txt,
                fill='black',
                font=FONT)