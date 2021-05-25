import summarizer as s

if __name__ == '__main__':
    txt = """ Dave Bautista, who will supposedly join the Guardians Of The Galaxy 3 cast as Drax The Destroyer for the last time, has shared one of the main reasons why he will be stepping down from the role. As per an article on ScreenRant, Dave Bautista, who had established himself in the entertainment world as a world champion wrestler, has cited his shirtless scenes as one of the reasons why he is not too inclined to play Drax anymore. Over the past few days, he has been making certain revelations and has been sharing his thoughts regarding Marvel Studios and his extraterrestrial character. Read on to know what else has he said.As reported earlier, Bautista, during his conversation with the officials at Collider, implied that the character, over the course of the four MCU films, has become too comedic for his own good. Bautista even went on to imply that the tragedy of Drax as a character, which involves his family being murdered by Ronan The Accuser, has been lost in the mix since Marvel went on to focus on several other characters that would be a part of the future superhero ensembles. However, he made it clear that the actor is, in no way, taking a jab at Marvel Studios as playing Drax essentially changed the trajectory of his life and he was very much aware of the direction that Marvel was headed in while taking on the part of Drax when it was offered to him.

As far as the Guardians of the Galaxy 3 release date in India is concerned, the film is scheduled for a worldwide release on 23rd May 2023. The film will see the return of the fan-favourite superhero team as well as introduce some new antagonists. Future Guardians Of The Galaxy 3 update(s) will be shared with the readers as and when they are made available. """
    r = 0.5
    triplets,sentences,processed = s.parsing(txt)
    summary = s.summarize(triplets,sentences,processed,r)
    cat,prob = s.predict_cat(txt)

    print(summary)
    print(cat)
    print(prob)

    
