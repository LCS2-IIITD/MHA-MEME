# MHA-Meme-Leveraging-Sentence-Demarcations-and-Multi-hop-Attention-for-Meme-Affect-Analysis
This repository contains the dataset and code for our ICWSM 2021 paper: 
[Exercise? I thought you said ‘Extra Fries’: Leveraging Sentence Demarcations and Multi-hop Attention for Meme Affect Analysis.](https://arxiv.org/abs/2103.12377) 

In this paper, we attempt to solve the three tasks suggested in the SemEval’20-Memotion Analysis competition. We propose a multi-hop attention-based deep neural network framework, called MHA-Meme, whose prime objective is to leverage the spatial-domain correspondence between the visual modality and various textual segments of an Internet meme to extract fine-grained feature representations for meme sentiment and affect classification. We evaluate MHA-Meme on the ‘Memotion Analysis’ dataset for all three sub-tasks - sentiment classification, affect classification, and affect class quantification. Our comparative study shows state-of-the-art performances of MHA-Meme for all three tasks compared to the top systems that participated in the competition. Unlike all the baselines which perform inconsistently across all three tasks, MHA-Meme outperforms baselines in all the tasks on average. Moreover, we validate the generalization of MHA-Meme on another set of manually annotated test samples and observe it to be consistent. Finally, we establish the interpretability of MHA-Meme.

# MHA-Meme Architecture

![](Images/MHA-Meme.png)

# Segmented Data Format

The foreground texts are critical in extracting the semantic level information from meme. However, they require special attention depending upon their position in the meme and
their reference to a specific region of the background image. Therefore, in the current work, we propose an attentive deep neural network architecture, called MHA-Meme (Multi-Hop
Attention for Meme Analysis), to carefully analyze the correspondence between the background image and each text segment at different spatial locations. To do so, at first, we
perform OCR (optical character recognition) to extract texts from the meme, and segment them into l sequence of text depending upon their spatial positions. Next, we process each
textual segment t<sub>i</sub> separately by establishing their correspondence with the background image I. The segmented text of each meme in the training ans test set is available at [dataset/train_splitted_all_tasks.csv](dataset/train_splitted_all_tasks.csv) and [dataset/test_splitted_all_tasks.csv](dataset/test_splitted_all_tasks.csv). Example format of segmented meme text with correspondng label:

| Image Name | Segment #1 | Segment #2 | Segment #3 | Segment #4 | Segment #5 | Segment #6 | Segment #7 | Segment #8 | Segment #9 | Segment #10 | Segment #11 | Segment #12 | Segment #13 | Segment #14 | Sentiment | Humor | Sarcasm | Offense | Motivation |
|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------|
| avengers_new-avenger-endgame-funny-memes-13.jpg |	YOU ARE BEAUTIFUL	| GO TO HELL | YOU ARE REALLY POWERFUL | GO TO HELL |	CAN YOU LEAD THE AVENGERS | REALLY? | GO TO HELL |0 |	0 |	0 |	0 |	0 |	0 |	0 |	positive |	funny	| not_sarcastic |	very_offensive |	motivational |

<p align="center"><img src="Images/avengers_new-avenger-endgame-funny-memes-13.jpg"></p>

# Citation
Please cite the following paper if you find this segmented dataset and MHA-Meme architecture useful in your research:

```
@misc{pramanick2021exercise,
      title={Exercise? I thought you said 'Extra Fries': Leveraging Sentence Demarcations and Multi-hop Attention for Meme Affect Analysis}, 
      author={Shraman Pramanick and Md Shad Akhtar and Tanmoy Chakraborty},
      year={2021},
      eprint={2103.12377},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

# Run the code

Clone the repository:

```
git clone https://github.com/ShramanPramanick/MHA-Meme-Affect-Analysis.git
cd MHA-Meme-Affect-Analysis
```

You will need python >= 3.6. Start by creating a virtual environment to run the code in:

```
python3 -m venv env
source env/bin/activate
```

Install all the requirements in the virtual environment:

```
pip install --upgrade pip
pip install -r requirements.txt
pip install bcolz
```
Train the model for Sentiment Classification:

```
python3 mha_meme_sentiment.py
```

Train the model for affect classification:

```
python3 mha_meme_affect.py
```
