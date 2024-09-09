Hi! If you're a musician interested in how AI can help your creativity, you've come to the right place! 
In this Repository, you will the code I wrote to come up with new melodic lines given snippets I took from recordings online.
I decided to do this project relating to Carnatic Music (Indian Classical). I took vocal snippets from the improvisational sections we call Alapanas.
Think of this as similar to a guitar solo in Classic Rock Songs. The artist gets to come up with their own melodies in the confines of a scale (what we call Raaga's)

In mp3_data_extraction+generation.py, you can load your own snippets (preferably 2-10 seconds) and train the model to understand the complexities, come up with its own melodic line,
and output a .pt file

Then, you can use mp3_conversion.py to convert that .pt file to an mp3 you can listen to!
