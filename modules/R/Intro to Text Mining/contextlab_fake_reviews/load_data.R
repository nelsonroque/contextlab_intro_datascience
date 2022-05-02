library(readr)
library(tidyverse)
library(tidytext)
library(textdata)
library(topicmodels)
library(wordcloud)
library(ggwordcloud)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# load full survey ----
# data source (https://osf.io/tyue9/)
df <- read_csv("fake reviews dataset.csv") %>%
  mutate(id = row_number()) %>% # add row id
  select(id, everything())

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# data quality report ----
df_dq = skimr::skim(df)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# text mining analyses ----
section_freeresponse_tokens = df %>%
  unnest_tokens(bigram,
                "text_",
                token = "ngrams",
                n = 2,
                drop = F) %>%
  separate(bigram, c("word1", "word2"), sep = " ", remove=F) %>%
  filter(!word1 %in% stop_words$word) %>%
  filter(!word2 %in% stop_words$word)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# tf_idf = statistic intended to reflect how important word is to a document
section_fr_1_tf_idf = section_freeresponse_tokens %>%
  count(category, word1) %>%
  bind_tf_idf(word1, category, n) %>%
  arrange(desc(tf_idf))

# vis tf_idf ----
section_fr_1_tf_idf %>%
  group_by(category) %>%
  slice_max(tf_idf, n = 5) %>%
  ungroup() %>%
  ggplot(aes(tf_idf, fct_reorder(word1, tf_idf), fill = category)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~category, ncol = 2, scales = "free") +
  labs(x = "tf-idf", y = NULL)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# wordcloud ---
# https://towardsdatascience.com/create-a-word-cloud-with-r-bde3e7422e8a
wordcloud::wordcloud(words = section_fr_1_tf_idf %>% pull(word1),
                     freq = section_fr_1_tf_idf %>% pull(n),
                     min.freq = 3,
                     max.words=200,
                     random.order=FALSE,
                     rot.per=0.1)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# sentiment analysis ----
AFINN <- get_sentiments("afinn")

# more options ---
# get_sentiments(lexicon = c("bing", "afinn", "loughran", "nrc"))

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# merge sentiment with dataset ----
sent_words <- section_freeresponse_tokens %>%
  inner_join(AFINN, by = c(word2 = "word"))

# produce various aggregates of sentiment ----
count_words_by_sent = sent_words %>%
  count(category, value, sort = TRUE)

ggplot(count_words_by_sent, aes(n)) +
  geom_histogram() +
  facet_grid(.~category)

avg_sent_by_category = sent_words %>%
 group_by(category) %>%
  summarise(avg_sent = mean(value, na.rm=T),
            sd_sent = sd(value, na.rm=T))

ggplot(avg_sent_by_category, aes(category, avg_sent)) +
  geom_bar(stat="identity")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# topic modeling analysis ----
# Resource: https://www.tidytextmining.com/topicmodeling.html

# ap_data = section_freeresponse_tokens %>%
#   cast_dtm(text_, word1, n)
# 
# ap_lda <- LDA(ap_data, k = 2, control = list(seed = 1234))
# ap_topics <- tidy(ap_lda, matrix = "beta")
# ap_documents <- tidy(ap_lda, matrix = "gamma")
# 
# ap_top_terms <- ap_topics %>%
#   filter(!is.na(term)) %>%
#   group_by(topic) %>%
#   slice_max(beta, n = 20) %>%
#   ungroup() %>%
#   arrange(topic, -beta)
# 
# ap_top_terms %>%
#   mutate(term = reorder_within(term, beta, topic)) %>%
#   ggplot(aes(beta, term, fill = factor(topic))) +
#   geom_col(show.legend = FALSE) +
#   facet_wrap(~ topic, scales = "free") +
#   scale_y_reordered()
# 
# beta_wide <- ap_topics %>%
#   mutate(topic = paste0("topic", topic)) %>%
#   pivot_wider(names_from = topic, values_from = beta) %>%
#   filter(topic1 > .001 | topic2 > .001) %>%
#   mutate(log_ratio_t2_t1 = log2(topic2 / topic1),
#          log_ratio_t3_t1 = log2(topic3 / topic1))
# 
# ggplot(beta_wide, aes(log_ratio_t2_t1, term)) +
#   geom_bar(stat="identity")