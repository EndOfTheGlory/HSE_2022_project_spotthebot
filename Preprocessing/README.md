###ENG

## Sequence of preprocessing
We have five steps, in which we divide our preprocessing. Using wikiextractor we recieved many pages of wiki in German with HTML. First, we need to:
1) Read our wiki files and compile all of the in one .csv file to continue.
Then we have one file with texts, but it's a mess, so we must clean the texts:
2) Cleaning text + stemming(to change all words to one iconic form)
We've got .csv file with clean texts, now we need to find the importance of word in context of documents(pages in wiki). For that we:
3) Create TF-IDF matrix.
4) We need SVD decommposition before proceeding
The last step is:
5) Vectorizing corpus of text to use them
In the end we have vectors, which we can subtitute  words with to teach our model.

###RU

##Последовательность предварительной обработки
У нас есть пять шагов, на которые мы разделяем нашу предварительную обработку. Используя wikiextractor, мы получили много страниц вики на немецком языке с HTML. Во-первых, нам необходимо:
1) Прочитать наши вики-файлы и скомпилировать их в один файл .csv, чтобы продолжить.
Тогда у нас есть один файл с текстами, но он малочитаем, поэтому мы должны очистить тексты:
2) Очистка текста + выделение корней (чтобы преобразовать все слова в одну иконическую форму)
У нас есть файл .csv с чистыми текстами, теперь нам нужно найти важность слова в контексте документов (страниц в вики). Для этого мы:
3) Создаем матрицу TF-IDF.
4) Нам нужна декомпозиция СВД, прежде чем продолжить
Последний шаг:
5) Векторизация корпуса текста для последующего использования
В итоге у нас есть векторы, которые мы можем заменить словами, чтобы обучить нашу модель.