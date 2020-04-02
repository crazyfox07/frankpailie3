# @Time    : 2020/4/1 8:11
# @Author  : lxw
# @Email   : liuxuewen@inspur.com
# @File    : crawl_pailie3.py
def pailie3():
    import requests
    from lxml import etree

    data = list()

    headers={
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.149 Safari/537.36',
        'Referer': 'http://datachart.500.com/pls/history/inc/history.php?limit=16039&start=04001&end=20040'

    }
    start = '04001'
    url = 'http://datachart.500.com/pls/history/inc/history.php?limit=16039&start={}&end=20040'.format(start)
    res = requests.get(url, headers=headers)
    html = etree.HTML(res.text)
    items = html.xpath('//tr[@class="t_tr1"]')
    for item in items:
        item_text = item.xpath('.//text()')
        print(item_text)
        num = item_text[1].replace(' ','')
        data.append(num)
    print(data)
    print(data[::-1])
    with open('pailie3.txt', 'w') as fw:
        for num in data[::-1]:
            fw.write('%s\n'%(num,))


if __name__ == '__main__':
    pailie3()