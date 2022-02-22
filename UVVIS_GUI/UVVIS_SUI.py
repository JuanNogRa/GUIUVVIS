import pygame
from io import BytesIO

def textTovoice(tts) :
    # convert to file-like object
            fp = BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            #--- play it ---
            pygame.init()
            pygame.mixer.init()
            pygame.mixer.music.load(fp)
            pygame.mixer.music.set_volume()
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)