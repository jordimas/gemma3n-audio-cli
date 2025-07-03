identify_lang_prompt = {
    "type": "text",
    "text": "Identify the languages of the audio and generate a JSON with the following fields\n"
    "- language: a iso-639 language identified\n"
    "- confidence level from 0 to 1 of the language identified\n",
}


transcribe_prompt_works = {
    "type": "text",
    "text": "You are given an audio recording in Catalan language with multiple speakers. Your task is to separate the speech segments by speaker and assign a unique speaker label to each segment. Output the start time, end time, and speaker ID for each segment."
    "Format:\n"
    "[start_time - end_time] Speaker_ID: Transcription (optional)\n"
    "\n"
    "Example:\n"
    "[00:00:00 - 00:00:05] Speaker_1: Hello, how are you?\n"
    "[00:00:06 - 00:00:10] Speaker_2: I'm good, thanks!\n"
    "\n"
    "Please provide the diarization results.\n",
}


# Does not seem to honor JSON
transcribe_prompt = {  # v2
    "type": "text",
    "text": "Please transcribe the following audio into accurate, fluent English. Focus on verbatim transcription, ensuring no content is added, omitted, or hallucinated. Do not transalate the transcription. Transcribe only what is spoken. If there are non-speech sounds like [laughter] or [noise], or if a word is incomprehensible, do not output any text. Do not repeat words unless they are repeated in the audio. Prioritize correct spelling and grammar in English. Maintain original phrasing and word order. Avoid repeated characters.",
}
