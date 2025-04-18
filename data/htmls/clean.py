from bs4 import BeautifulSoup
import requests
import re

def readLinks(fileName):
	links = []
	with open(fileName, "r") as f:
		links = [line.strip() for line in f if line.strip()]

	return links

def pullWebsite(url):
	response = requests.get(url)
	if response.status_code == 200:
		return response.text
		
	else:
		print("Failed to fetch page:", response.statusCode)
		return None

def extract_text_sections_with_tables(delimiter='|||'):

	links = readLinks("links.txt")

	for i, link in enumerate(links):

		website = pullWebsite(link)

		soup = BeautifulSoup(website, 'html.parser')

		sections = []
		current_section = ""

		# Traverse through all relevant tags in document order
		for tag in soup.find_all(['h4', 'h5', 'p', 'table']):
			if tag.name in ['h4', 'h5']:
				if current_section:
					sections.append(current_section.strip())
					current_section = ""
				current_section += tag.get_text(strip=True) + "\n"

			elif tag.name == 'p':
				current_section += tag.get_text(" ", strip=True) + "\n"

			elif tag.name == 'table':
				caption = tag.caption.get_text(strip=True) if tag.caption else "Table"
				current_section += f"\n{caption}:\n"

				# Extract headers
				headers = []
				thead = tag.find('thead')
				if thead:
					header_cells = thead.find_all('th')
					headers = [cell.get_text(" ", strip=True) for cell in header_cells]
					current_section += ' | '.join(headers) + "\n"

				# Extract rows
				for row in tag.find_all('tr'):
					cells = row.find_all(['td'])
				if cells:
					row_text = ' | '.join(cell.get_text(" ", strip=True) for cell in cells)
					current_section += row_text + "\n"

				if current_section:
					sections.append(current_section.strip())

				final_text = f"\n{delimiter}\n".join(sections)

				with open(str(i) + ".txt", 'w', encoding='utf-8') as out:
					out.write(final_text)

if __name__ == "__main__":
	text = extract_text_sections_with_tables()
	print("Done! Extracted sections and formatted tables.")

