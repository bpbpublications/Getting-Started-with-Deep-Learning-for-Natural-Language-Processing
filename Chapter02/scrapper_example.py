# Importing required packages
from pprint import pprint

import requests
from bs4 import BeautifulSoup

# Getting page Content
page = requests.get('https://www.carwale.com/mahindra-cars/marazzo/userreviews/', verify=False)
contents = page.content
# parsing DOM using Soup
soup = BeautifulSoup(contents, 'html.parser')
# getting all data under userReviewListing
mydivs = soup.findAll(id="userReviewListing")
# getting All DIV under userReviewListing
for i in mydivs:
    for j in i.find_all("div"):  # Getting individual DIV under previous DIV
        try:
            m, n = j.find_all("span")  # Getting SPAN under DIV  One span is having rating info one SPAN is full Comment
            pprint({"Rating": m.get('class', []), "Full_Comment": n.text})  # Making Dict for Rating and Full Text
        except:
            ""
