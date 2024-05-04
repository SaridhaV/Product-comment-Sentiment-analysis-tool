import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog, messagebox, Label, Button, Text
import csv
from wordcloud import WordCloud
import colorsys

class ProductSentimentAnalysisTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Product Comment Sentiment Analysis Tool")
        self.root.geometry("1000x400")

        self.label = Label(root, text="Select CSV file:")
        self.label.pack()

        self.selected_file_label = Label(root, text="")
        self.selected_file_label.pack()

        self.button = Button(root, text="Upload CSV", command=self.upload_csv)
        self.button.pack()

        self.text_area = Text(root, height=20, width=50)
        self.text_area.pack(side="right", fill="y")

    def upload_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])

        if not file_path:
            messagebox.showerror("Error", "No file selected.")
            return

        try:
            df = pd.read_csv(file_path, encoding='utf-8', quoting=csv.QUOTE_ALL)
            self.analyze_sentiment(df)
            self.selected_file_label.config(text=f"Selected file: {file_path}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def analyze_sentiment(self, df):
        try:
            sia = SentimentIntensityAnalyzer()
            df['Sentiment'] = df['Comment'].apply(lambda x: sia.polarity_scores(x)['compound'])

            # Classify sentiment
            df['Sentiment'] = df['Sentiment'].apply(lambda score: 'positive' if score > 0 else ('negative' if score < 0 else 'neutral'))

            # Generate word cloud with sentiment coloring
            self.generate_word_cloud(df)

            # Display sentences and summary statistics
            self.display_sentences(df)
            self.display_summary(df)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def generate_word_cloud(self, df):
        # Prepare text for word cloud
        text = ' '.join(df['Comment'])

        # Generate word cloud
        wordcloud = WordCloud(width=400, height=400, background_color='white').generate(text)

        # Color words based on sentiment
        sentiment_colors = {
            'positive': 'green',
            'negative': 'red',
            'neutral': 'gray'
        }
        sentiment_values = df['Sentiment'].value_counts(normalize=True).to_dict()
        for word, sentiment in sentiment_values.items():
            if word in wordcloud.words_:
                h, s, l = colorsys.rgb_to_hls(*colorsys.hex_to_rgb(sentiment_colors[word]))
                wordcloud.recolor(word, rgb=colorsys.hls_to_rgb(h, l, s * sentiment))

        # Plot word cloud
        plt.subplot(1, 3, 1)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud with Sentiment Coloring')

    def display_sentences(self, df):
        sentences = '\n'.join(df['Comment'].values.tolist())
        self.text_area.delete(1.0, "end-1c")
        self.text_area.insert("end", sentences)

    def display_summary(self, df):
        sentiment_counts = df['Sentiment'].value_counts()

        total_responses = len(df)
        total_snippets = len(df['Comment'].str.split().explode())

        summary_text = f"Total Responses: {total_responses}\nTotal Snippets: {total_snippets}\n\nSentiment Distribution:\n{sentiment_counts}"
        label_summary = Label(self.root, text=summary_text)
        label_summary.pack()

        # Plot pie chart for sentiment distribution
        plt.subplot(1, 3, 2)
        plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140)
        plt.title('Sentiment Distribution')

        # Plot bar chart for sentiment counts
        plt.subplot(1, 3, 3)
        sentiment_counts = sentiment_counts.reindex(['positive', 'negative', 'neutral']).fillna(0)
        plt.bar(sentiment_counts.index, sentiment_counts.values, color=['green', 'red', 'gray'])
        plt.title('Sentiment Counts')

        plt.show()

if __name__ == "__main__":
    nltk.download('vader_lexicon')  # Manually download the vader_lexicon data
    root = Tk()
    app = ProductSentimentAnalysisTool(root)
    root.mainloop()
