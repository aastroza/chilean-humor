from loguru import logger
import json

from chilean_humor.utils import extract_routines_ids
from chilean_humor.joke import Joke, fuse_jokes
from chilean_humor.refine import detect_continuity


def main():
    routine_ids = extract_routines_ids(folder='jokes')

    for routine_id in routine_ids:
        filename = f"jokes/routine_{routine_id}_repertoire.jsonl"
        jokes = []
        logger.info(f"Refining jokes for routine {routine_id}.")
        with open(filename, "r", encoding="utf-8") as file:
            for line in file:
                dict_line = json.loads(line)

                if len(jokes) == 0:
                    jokes.append(Joke(**dict_line))
                    continue

                else:
                    joke = Joke(**dict_line)
                    previous_joke = jokes[-1]
                    
                    analysis = detect_continuity(previous_joke.corrected_transcript, joke.corrected_transcript)
                    if analysis.outcome.value == "continuation":
                        logger.info("Fusing jokes.")
                        fused_joke = fuse_jokes(previous_joke, joke)
                        jokes[-1] = fused_joke
                    else:
                        logger.info("Adding new joke.")
                        jokes.append(joke)

        logger.info(f"Writing refined jokes for routine {routine_id}.")               
        for joke in jokes:
            try:
                with open(f"jokes_refined/routine_{routine_id}_refined_repertoire.jsonl", "a", encoding="utf-8") as file:
                    json_line = joke.json()
                    file.write(json_line + "\n")
            except Exception as e:
                logger.info(f"Error writing joke for {routine_id} {e}")
                continue

if __name__ == "__main__":
    main()