"""
Purpose: Scrape dữ liệu từ từng trang của mỗi khóa học trên Coursera
"""

import pandas as pd 
import requests
from bs4 import BeautifulSoup
import os


class DataHunter:
	"""
	To make the big new dataset
	"""

	df = None # dataframe from scraper.py
	skills = []
	about = []
	new_career_starts = []
	pay_increase_prom = []
	estimate_toc = []
	instructors = []

	def __init__(self, df):
		"""
        Khởi tạo với một DataFrame từ file csv
        """
		self.df = df

	def scrape_features(self, page_url):
		"""
		Scrapes features from each page
		-----
		page_url:
			URL of the page
		"""

		# create the soup with a certain page URL
		course_page = requests.get(page_url)
		course_soup = BeautifulSoup(course_page.content,
										'html.parser')
		# pick course skills
		try:
			cskills = course_soup.find_all("span", "_x4x75x")
			temp = ""
			for idx in range(len(cskills)):
				temp = temp + cskills[idx].text
				if(idx != len(cskills)-1):
					temp = temp + ","
			self.skills.append(temp)
		except:
			self.skills.append("Missing")

		# pick about course
		try:
			# lấy nội dung của một component HTML dựa trên tên class.
			cdescr = course_soup.select(".description")
			self.about.append(cdescr[0].text)
		except:
			self.about.append("Missing")

		# pick learner stats
		try:
			learn_stats = course_soup.select(
				"._1qfi0x77 > .LearnerOutcomes__text-wrapper > .LearnerOutcomes__percent" 
			)
		except:
			pass
		try:
			self.new_career_starts.append((float(learn_stats[0].text.replace('%',''))))
		except:
			self.new_career_starts.append("Missing")
		try:
			self.pay_increase_prom.append((float(learn_stats[1].text.replace('%',''))))
		except:
			self.pay_increase_prom.append("Missing")

		# pick estimated time to complete
		try:
			# lấy nội dung của một component HTML dựa trên tên class.
			props = course_soup.select("._16ni8zai")

			done = 0 # biến đếm để ngăn chặn các giá trị trùng lặp

			# Khởi tạo biến etoc với giá trị "Missing", sẽ cập nhật giá trị này nếu tìm thấy nội dung cần tìm trong các phần tử props
			etoc = "Missing"
			
			# duyệt qua từng phần tử trong props
			for idx in range(len(props)):

				# Ktra chuỗi 'to complete' có xuất hiện trong nội dung text
				# của phần tử thứ idx không và xem điều kiện done chưa bằng 0
				if('to complete' in props[idx].text and done==0):
					# Nếu điều kiện trên là đúng, cập nhật etoc bằng nội dung text của phần tử thứ idx
					etoc = props[idx].text
					done+=1

			# Khi đã duyệt qua hết tất cả các phần tử trong props, thì thêm giá trị của etoc vào self.estimate_toc
			self.estimate_toc.append(etoc)
		except:
			# Nếu không tìm thấy chuỗi "to complete" ở bất cứ đâu, etoc sẽ giữ nguyên giá trị "Missing", và "Missing" sẽ được thêm vào self.estimate_toc
			self.estimate_toc.append("Missing")

		# pick instructors
		try:
			# lấy nội dung của một component HTML dựa trên tên class.
			instructors = course_soup.select(".instructor-name")
			temp=""
			for idx in range(len(instructors)):
				temp = temp + instructors[idx].text
				if(idx != len(instructors)-1):
					temp = temp + ","
			self.instructors.append(temp)
		except:
			self.instructors.append("Missing")

	def extract_url(self):
		"""
		Extracts URLs from the dataframe loaded
		"""

		for url in self.df['Course URL']:
			self.scrape_features(url)

	def make_dataset(self):
		"""
		Tạo bộ dataset mới từ thông tin vừa mới thu được
		"""

		# gọi extract_url() để bắt đầu quá trình thu thập dữ liệu
		self.extract_url()

		# Tạo DataFrame từ dữ liệu vừa thu được
		data_dict = {
				"Skills":self.skills,
				"Description":self.about,
				"Percentage of new career starts":self.new_career_starts,
				"Percentage of pay increase or promotion":self.pay_increase_prom,
				"Estimated Time to Complete":self.estimate_toc,
				"Instructors":self.instructors
			}

		data = pd.DataFrame(data_dict)

		return data

def main():

	# Đọc Dataframe từ file csv
	source_path = os.path.join("data/coursera-courses-overview.csv")
	df = pd.read_csv(source_path)
	
	# Tạo một instance của DataHunter và đi qua từng trang thông qua URL trong DataFrame
	dh = DataHunter(df)
	df = dh.make_dataset()

	# Lưu DataFrame vào một file CSV mới
	destination_path = os.path.join("data/coursera-individual-courses.csv")
	df.to_csv(destination_path, index=False)

# Khi script này được chạy độc lập, main() sẽ được gọi.
if __name__=="__main__":
	main()