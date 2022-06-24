import os


fields = ['1pNPRspk0RyvI6c1jCaTknaL2GYDB1qOZ',  #0
          '16sSK7LNqzOZPux_IKhCcbhkqIxKKyl8t',  #1
          '1q3CvOhPnR5FKcnNXqyg4baIjZ050uSeG',  #3
          '1gV7H8pHyoBDVi6BNqGecLDRACISaQiWj',  #4
          '1yn4kpvjXL2IPYu3RTYkFU1RY8KSGpu1Z',  #5
          '1sVhjh0XR4k_Ja34aC_EiGVYs8mRfx9aL',  #6
          '12N1cOQENS0_S1azBUPv4OcOp0HyIO8bR',  #7
          '1O5tDHNeTopnjRTTNrSnC2wX0TBVplk5J',  #8
          '1lNu8Zlrmo0Q5ZZhhAdFB2KvXBIAolScr',  #9
          '18B5WDY9Hks1-7UnbmMLLHnQoEz50Fohp',  #10
          '1L2NvoRTLPyIEWAYZ-Ib3KAwl0utLoTdh',  #11
          '1g22NJP6yOa8nqs5q3Y6VQ79DQkcIZV7F',  #12
          '1UAER0cefEg_nyQmKeZESWWol6laUs7Ji',  #13
          '1f4uDFTaC3MXqIr65-lAPrib5z7JnokGQ',  #14
          '1ZimJb3yuP_W-6ZPguG-zjBGma2Suvfxz',  #15
          '10nXsZZ5RBbLS8OWyidMiI_SIRjqIWvLk',  #16
          '1jV5r9Cf7sPk7Eq5AX8gabDgv6d5wNwd-',  #17
          '1za4hBkE26nOCxcSemkpT2HZyrZ4zLJlu',  #18
          '1dqbCZITv21zdOg4XKnSXgU3NNMMtTLwK',  #19
          '1V9YU9HF9aCin_vMYfg-w-DE_DcsOwqrI',  #20
          '1ze9OanQ8WXJeNy7FpyM0my13oNLdKTbP',  #21
          '1iBED5rEgcHuNksyZmxJk72fycjR5Oe6U',  #22
          '1QTcwHtw_DcnKSDy0XrBngp00_KtOZ3m1',  #23
          '1IpSW05EkSlOSOF-4GGNuYufTkgnSIirS',  #24
          '1SAGDo7Gl3GLO3E1FKD1XUqZPO_m2FsXT',  #25
          '1viTiiNIV50rkDlelAD0MxsaTXnYgQpRT',  #26
          '1qARoShywjKV1G0Oq-6hSOMN3-luipJcb',  #27
          '1DYomYydYQhYzeFigeBtoUW8Vtp8Hvx0Q',  #28
          '1kTyyqjsc2A6GvxCV9Znf_pAu0p41c1IK',  #29
          '1TdLUkjXjtaAyqKXvcAtXo72lccxGDYAn',  #30
          '1Gr8EC2-xtedFEvYhSU3eCCySl_zTvCj2',  #31
          '1QL5swAGLBE4Rc-YucBqhY7RmOzhJ_WAs',  #32
          '1WuMK97WRXIdA2GGvB3zLW93Ncsydjigp',  #33
          '1AjhUKpAIviRhy0biGJSRd26huvaGjpV3',  #34
          '18ZzCIOOSptmf7hBrj_dnRfFzMHTBUgAh',  #35
          '174U1EEU7gVX3PQTp8hb6nJMk67iXY076',  #36
          '1LR2wc0zuosaAI5lS6JtCROAok0HxRNNc',  #37
          '1Xt43Jd67RgKYC463NnnJa0EpzxIRY1kc',  #38
          '1D82PQUy5MKxrxrVCCAzzql04OJRgJChT',  #39
          ]

download_list = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38]

for i in download_list:
    os.system(f'/home/ec2-user/gdrive download {fields[i]}')


