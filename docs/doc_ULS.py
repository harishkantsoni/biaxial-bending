

# This module creates the documentation of the ULS analysis as Markdown text. The text is saved in a long string
# and exported to a Markdown file (.md)

# Variables from calculations to use in documentation
name = 'Tim'
P = 485
Mx = 2315
My = 3151

# Documentation string
string = f'''
<header>
ULS Documentation
============
</header>

Table of contents
===========
[TOC]

## Heading 2

### Heading 3

#### Heading 4

##### Heading 5

$$ x^2 $$

$$ (P, M_x, M_y) = ({P} \\textrm{{ kN}}, {Mx} \\textrm{{ kNm}}, {My}\\textrm{{ kNm}})$$


Inline math: ${P}^2 = {P**2}$

My name is {name}


'''

print(string)


# Write string to Markdown file
file = open('doc_ULS.md','w')
file.write(string)
file.close()
