# MHA-Meme-Leveraging-Sentence-Demarcations-and-Multi-hop-Attention-for-Meme-Affect-Analysis
This repository contains the dataset and code for our ICWSM 2021 paper: Exercise? I thought you said ‘Extra Fries’: Leveraging Sentence Demarcations and Multi-hop Attention for Meme Affect Analysis. In this paper, we attempt to solve the three tasks suggested in the SemEval’20-Memotion Analysis competition. We propose a multi-hop attention-based deep neural network framework, called MHA-Meme, whose prime objective is to leverage the spatial-domain correspondence between the visual modality and various textual segments of an Internet meme to extract fine-grained feature representations for meme sentiment and affect classification. We evaluate MHA-Meme on the ‘Memotion Analysis’ dataset for all three sub-tasks - sentiment classification, affect classification, and affect class quantification. Our comparative study shows state-of-the-art performances of MHA-Meme for all three tasks compared to the top systems that participated in the competition. Unlike all the baselines which perform inconsistently across all three tasks, MHA-Meme outperforms baselines in all the tasks on average. Moreover, we validate the generalization of MHA-Meme on another set of manually annotated test samples and observe it to be consistent. Finally, we establish the interpretability of MHA-Meme.

# MHA-Meme Architecture

![](Images/MHA-Meme.png)

# Segmented Data Format

The foreground texts are critical in extracting the semantic level information from meme. However, they require special attention depending upon their position in the meme and
their reference to a specific region of the background image. Therefore, in the current work, we propose an attentive deep neural network architecture, called MHA-Meme (Multi-Hop
Attention for Meme Analysis), to carefully analyze the correspondence between the background image and each text segment at different spatial locations. To do so, at first, we
perform OCR (optical character recognition) to extract texts from the meme, and segment them into l sequence of text depending upon their spatial positions. Next, we process each
textual segment t<sub>i</sub> separately by establishing their correspondence with the background image I. The segmented text of each meme in the training ans test set is available at [dataset/train_splitted_all_tasks.csv](dataset/train_splitted_all_tasks.csv) and [dataset/test_splitted_all_tasks.csv](dataset/test_splitted_all_tasks.csv). Example format of segmented meme text with correspondng label:

| avengers_new-avenger-endgame-funny-memes-13.jpg |	YOU ARE BEAUTIFUL	GO TO HELL | YOU ARE REALLY POWERFUL | GO TO HELL |	CAN YOU LEAD THE AVENGERS | REALLY? | GO TO HELL |0 |	0 |	0 |	0 |	0 |	0 |	0 |	positive |	funny	| not_sarcastic |	very_offensive |	motivational |
![image](https://user-images.githubusercontent.com/40575004/111927180-2f5ff280-8a86-11eb-9bcd-8856074bffb5.png)

