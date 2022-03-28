##############################################################################
#All this code below is to scrape slaray data from the ESPN Data Base for NBA salary in =2018-2019
install.packages("rvest")
install.packages("writexl")
install.packages("tidyverse")
install.packages("stringr")
library(rvest) # web scraping
library(tidyverse) # pipe operators
library(stringr) # data cleanup
library(writexl)
# construct urls
season_urls = sprintf('http://www.espn.com/nba/salaries/_/year/%s/', 2019)
season_urls = c(season_urls, 'http://www.espn.com/nba/salaries/_/') # add current year

# compile data
datalist = list()
n = 1
for (i in 1:length(season_urls)) {
  Sys.sleep(.2)
  
  # download 
  page = read_html(season_urls[i])
  
  # determine number of sub-pages
  text = page %>% html_node('.page-numbers') %>% html_text() 
  num_pages = text %>% str_sub(., start= -2) %>% as.numeric()
  print(paste(i, " : ", text, " : ", num_pages, sep = ''))
  
  # dynamically create list of sub-page urls
  temp_urls = paste(season_urls[i], 'page/', 2:num_pages, sep = '')
  temp_urls = c(temp_urls, season_urls[i]) # add first page back in
  
  # loop through sub-pages
  for (j in 1:length(temp_urls)) {
    print(temp_urls[j])
    Sys.sleep(2)
    
    # determine year
    year = str_sub(season_urls[i], -5, -2)
    page = read_html(temp_urls[j])
    table = page %>% html_table()
    datalist[[n]] = table[[1]] %>% mutate(year = year, page = j)
    n = n + 1
  }
}

# combine data
raw = do.call('rbind', datalist)

# clean data
salaries = raw %>%
  select(rank = X1, name = X2, team = X3, salary = X4, season = year) %>%
  filter(rank != 'RK') %>%
  mutate(salary = as.numeric(gsub('\\$|,', '', salary))) %>% 
  separate(name, into = c('name', 'position'), sep = ',', remove = T) %>%
  mutate(rank = as.numeric(rank)) %>% arrange(season, rank) %>%
  mutate(season = ifelse(season == 'es/_', 2020, season))

# export data
#setwd("Users\eugen\Downloads")
#write_csv(salaries, 'nba-salaries.csv')
write_xlsx(salaries,"C:\\Users\\eugen\\Documents\\Salaries.csv")

