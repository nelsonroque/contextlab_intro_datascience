library(readr)
library(tidyverse)
library(tidytext)
library(textdata)
library(topicmodels)
library(wordcloud)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# load full survey ----
# data source (https://osf.io/tyue9/)
df <- read_csv("fake reviews dataset.csv", skip = 1) %>%
  mutate(id = row_number()) %>%
  select(id, everything()) %>%
  ruf::make_tidy_colnames(.)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# data quality report ----
df_dq = skimr::skim(df)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

section_freeresponse = df %>%
  select(id,
         contains("there_anything_else"),
         contains("please_tell_us_any_factors"),
         contains("teaching_strategies_used")) %>%
  pivot_longer(cols = is_there_anything_else_you_would_like_to_tell_the_ccoe_faculty_and_administration_about_how_you_can_be_better_supported_either_personally_or_academically_what_changes_to_remote_instruction_would_improve_your_remote_learning_experience:if_any_other_teaching_strategies_used_by_your_instructors_have_also_positively_impacted_your_remote_learning_experience_please_add_them_here,
               names_to = "prompt", values_to = "free_response") %>%
  mutate(prompt_s = recode(prompt,
                           `if_any_other_teaching_strategies_used_by_your_instructors_have_also_positively_impacted_your_remote_learning_experience_please_add_them_here` = "teaching_strategies",
                           `is_there_anything_else_you_would_like_to_tell_the_ccoe_faculty_and_administration_about_how_you_can_be_better_supported_either_personally_or_academically_what_changes_to_remote_instruction_would_improve_your_remote_learning_experience` = "ccoe_faculty_support",
                           `please_tell_us_any_factors_that_reduced_or_limited_your_ability_to_perform_well_during_remote_teaching` = "factors_reduce_ability"))

# text mining analyses ----

section_freeresponse_tokens = section_freeresponse %>%
  unnest_tokens(bigram,
                "free_response",
                token = "ngrams",
                n = 2,
                drop = F) %>%
  separate(bigram, c("word1", "word2"), sep = " ", remove=F) %>%
  filter(!word1 %in% stop_words$word) %>%
  filter(!word2 %in% stop_words$word)

# tf_idf = statistic intended to reflect how important word is to a document
section_fr_1_tf_idf = section_freeresponse_tokens %>%
  count(prompt_s, word1) %>%
  bind_tf_idf(word1, prompt_s, n) %>%
  arrange(desc(tf_idf))

# wordcloud ---
# https://towardsdatascience.com/create-a-word-cloud-with-r-bde3e7422e8a
wordcloud::wordcloud(words = section_fr_1_tf_idf %>% filter(prompt_s == "factors_reduce_ability") %>% pull(word1),
                     freq = section_fr_1_tf_idf %>% filter(prompt_s == "factors_reduce_ability") %>% pull(n),
                     min.freq = 3,
                     max.words=200,
                     random.order=FALSE,
                     rot.per=0.1)

wordcloud::wordcloud(words = section_fr_1_tf_idf %>% filter(prompt_s != "factors_reduce_ability") %>% pull(word1),
                         freq = section_fr_1_tf_idf %>% filter(prompt_s != "factors_reduce_ability") %>% pull(n),
                         min.freq = 2,
                         max.words=200,
                         random.order=FALSE,
                         rot.per=0.1)

library(ggwordcloud)
set.seed(1234) # for reproducibility
ggplot(section_fr_1_tf_idf, aes(label = word1, size = n)) +
  geom_text_wordcloud() +
  scale_size_area(max_size = 40) +
  theme_minimal()

cowplot::plot_grid(a,b)

# vis tf_idf ----
section_fr_1_tf_idf %>%
  group_by(prompt_s) %>%
  slice_max(tf_idf, n = 5) %>%
  ungroup() %>%
  ggplot(aes(tf_idf, fct_reorder(word1, tf_idf), fill = prompt_s)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~prompt_s, ncol = 2, scales = "free") +
  labs(x = "tf-idf", y = NULL)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# topic modeling analysis ----
# Resource: https://www.tidytextmining.com/topicmodeling.html

ap_data = section_fr_1_tf_idf %>%
  cast_dtm(prompt_s, word1, n)
ap_lda <- LDA(ap_data, k = 2, control = list(seed = 1234))
ap_topics <- tidy(ap_lda, matrix = "beta")
ap_documents <- tidy(ap_lda, matrix = "gamma")

ap_top_terms <- ap_topics %>%
  filter(!is.na(term)) %>%
  group_by(topic) %>%
  slice_max(beta, n = 20) %>%
  ungroup() %>%
  arrange(topic, -beta)

ap_top_terms %>%
  mutate(term = reorder_within(term, beta, topic)) %>%
  ggplot(aes(beta, term, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  scale_y_reordered()

beta_wide <- ap_topics %>%
  mutate(topic = paste0("topic", topic)) %>%
  pivot_wider(names_from = topic, values_from = beta) %>%
  filter(topic1 > .001 | topic2 > .001) %>%
  mutate(log_ratio_t2_t1 = log2(topic2 / topic1),
         log_ratio_t3_t1 = log2(topic3 / topic1))

ggplot(beta_wide, aes(log_ratio_t2_t1, term)) +
  geom_bar(stat="identity")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# sentiment analysis ----
AFINN <- get_sentiments("afinn")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# motivation

# not_words <- section_fr_1_bigram %>%
#   filter(word1 %in% c("anxious")) %>%
#   inner_join(AFINN, by = c(word2 = "word")) %>%
#   count(word2, value, sort = TRUE)
