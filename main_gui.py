from tkinter import *
from Predictions_code import *
from PIL import Image,ImageTk
import os
global t
t = 0

root = Tk()
root.title('Stock Projection')
root.minsize(400,100)


def model_exist(m):
    n = str(existing_data(m))
    n = "Tommorow's change = " + n
    print(n)

    result_window = Toplevel(root)

    result_window.minsize(1500,600)

    canvas1= Canvas(result_window, width= 600, height= 410)
    canvas1.grid(row=0,column=0)
    img1= (Image.open("models/"+m+"/time_series/test_figure.png"))
    resized_image1= img1.resize((600,410), Image.ANTIALIAS)
    new_image1= ImageTk.PhotoImage(resized_image1)
    canvas1.create_image(10,10, anchor=NW, image=new_image1)
    label1 = Label(result_window,text = "Testing Figure and error from Time series analysis").grid(row=1,column=0)

    canvas2= Canvas(result_window, width= 600, height= 410)
    canvas2.grid(row=0,column=1)
    img2= (Image.open("models/"+m+"/time_series/projected.png"))
    resized_image2= img2.resize((600,410), Image.ANTIALIAS)
    new_image2= ImageTk.PhotoImage(resized_image2)
    canvas2.create_image(10,10, anchor=NW, image=new_image2)
    label2 = Label(result_window,text = "Projected Opening and Closing of Stock price").grid(row=1,column=1)

    canvas3= Canvas(result_window, width= 600, height= 410)
    canvas3.grid(row=0,column=2)
    img3= (Image.open("models/"+m+"/regression/model_analysis.png"))
    resized_image3= img3.resize((600,410), Image.ANTIALIAS)
    new_image3= ImageTk.PhotoImage(resized_image3)
    canvas3.create_image(10,10, anchor=NW, image=new_image3)
    label3 = Label(result_window,text = "Error and testing of regression model for stock change").grid(row=1,column=2)

    label4 = Label(result_window,text = n,font=50,bg='black',fg='white').grid(row=2,column=1,padx=10,pady=10)

    result_window.mainloop()

    
def model_create(m):
    new_data(m)
    model_exist(m)

global button

def other_option():
    label.config( text = e.get()) 
    model_create(e.get())
e = Entry(root,)
e.insert(END,string="Enter Stock Ticker")
button = Button(root,text = "Enter Name" , command = other_option )

global other
root.geometry( "200x200" )

def show(event):
    global t
    if clicked.get() == 'Other':
        if(t==0):
            t+=1
            e.pack()
            button.pack()
    else:
        if(t!=0):
            e.pack_forget()
            button.pack_forget()
        t=0
        
        label.config( text = clicked.get())     
        model_exist(clicked.get()) 
  
clicked = StringVar()
  
clicked.set('Select Stock Model')
  
options = ((os.listdir('models/'))+['Other'])[1:]

drop = OptionMenu( root , clicked , *options, command=show )
drop.pack()

label = Label( root , text = " " )
label.pack()
  
root.mainloop()
