from tkinter import *
from Predictions_code import *
import os
global t
t = 0

root = Tk()

def model_exist(m):
    existing_data(m)
    
def model_create(m):
    new_data(m)
    existing_data(m)

global button

def other_option():
    label.config( text = e.get()) 
    model_create(e.get())
e = Entry(root)
button = Button(root,text = "click Me" , command = other_option )

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
  
options = ((os.listdir('models/'))+['Other'])

drop = OptionMenu( root , clicked , *options, command=show )
drop.pack()

label = Label( root , text = " " )
label.pack()
  
root.mainloop()
