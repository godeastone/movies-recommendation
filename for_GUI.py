from tkinter import *
import Project

win = Tk()

win.geometry("1000x500")
win.title("Movie recommend system")
win.option_add("*Font", "맑은고딕 15")

label1 = Label(win,text = "Please enter your favorite 5 movies.", padx=20, pady=10)
label1.pack(padx=20, pady=20)

ent = Entry(win, width=800)
ent.pack(padx=20, pady=10)

text=Text(win, width=800)


def get_movies():
    movies = ent.get()
    movie_list2 = []
    movie_list1 = movies.split(',')
    for movie in movie_list1:
        movie = movie.strip()
        movie_list2.append(movie)

    # recommendation algorithm
    result = Project.main(movie_list2)

    text.insert(CURRENT, result)
    text.pack(padx=20, pady=20)
    return

btn = Button(win, text = "Recommend",height = 2, width = 12)
btn.config(command = get_movies)
btn.pack()

win.mainloop()