"""Generates a synthetic, labelled comment dataset (data/comments.csv).
NOTE: synthetic demo data. Replace data/comments.csv with a real dataset
(e.g. Kaggle 'Cyberbullying Classification' / Jigsaw Toxic Comments) with
columns: text,label (1 = bullying, 0 = not bullying) for real-world results."""
import random, csv
random.seed(42)

bully_t = [
 "you are such a {neg}, nobody likes you", "{neg}! just go away and never come back",
 "what a {neg}, you should be ashamed of yourself", "everyone hates you, you {neg}",
 "you look so {look}, disgusting", "i hope you {harm}, you {neg}",
 "shut up you {neg}, no one asked for your opinion", "you're a {neg} and you will always be a loser",
 "nobody wants you here, go {harm}", "stop posting, you {neg}, you make me sick",
 "you are {look} and {look2}, delete your account", "such a {neg}, your whole family must be embarrassed of you",
 "lol look at this {neg}, so {look}", "you are worthless, the world is better without {neg}s like you",
 "get lost {neg}, we all laugh at you behind your back", "imagine being this {look}. pathetic {neg}",
 "i will make your life hell, {neg}", "no one will ever love a {look} {neg} like you",
 "you {neg}, you can't do anything right", "everybody in class is making fun of you, {neg}",
]
neg = ["idiot","loser","moron","stupid freak","dumb clown","retard","pathetic fool","useless trash","creep","failure","ugly pig","worthless waste"]
look = ["ugly","fat","disgusting","hideous","pathetic","gross","stupid-looking"]
look2 = ["dumb","annoying","worthless","creepy"]
harm = ["disappear","die","rot","vanish forever","get lost","fail everything"]

safe_t = [
 "great work on this {thing}, really {pos}!", "i totally disagree with this {thing}, but thanks for sharing",
 "this {thing} is {pos}, keep it up", "can anyone help me with this {thing}?",
 "congratulations on your {thing}, you deserve it", "thanks for the {thing}, it was really helpful",
 "you killed it in that {thing}, absolutely {pos}", "this song is killing me, so {pos}",
 "what time does the {thing} start tomorrow?", "i love how {pos} this {thing} turned out",
 "that was a stupid mistake on my part, sorry everyone", "this exam is hell but we will get through it together",
 "i hate Mondays, honestly", "the weather is terrible today, stay safe", "your {thing} made my day, {pos}!",
 "i'm so proud of you for the {thing}", "well played, the {thing} was {pos}", "let's meet at the library for the {thing}",
 "this bug is so dumb, spent hours fixing my own {thing}", "he is not an idiot, please stop saying that, be kind",
 "that movie was so bad it was funny, still enjoyed the {thing}", "anyone else find this {thing} a bit boring?",
 "please don't call people names, it hurts", "stop bullying others online, it's not okay",
]
thing = ["project","video","presentation","match","assignment","design","post","photo","performance","code","album","event"]
pos = ["amazing","awesome","brilliant","fantastic","inspiring","wonderful","excellent","impressive"]

def noise(s):
    r = random.random()
    if r < .15: s = s.upper()
    elif r < .30: s = s.replace("a","@").replace("i","1") if random.random()<.5 else s.replace("o","0")
    elif r < .40: s = s + random.choice(["!!!","...","??"," lol"," \U0001F621"," \U0001F602"])
    elif r < .48: s = s.replace(" ", "  ", 1)
    return s

rows=set()
while len(rows) < 3000:
    if random.random()<.5:
        gi=random.randrange(len(bully_t)); t=bully_t[gi].format(neg=random.choice(neg),look=random.choice(look),look2=random.choice(look2),harm=random.choice(harm)); y=1; g="b%d"%gi
    else:
        gi=random.randrange(len(safe_t)); t=safe_t[gi].format(thing=random.choice(thing),pos=random.choice(pos)); y=0; g="s%d"%gi
    rows.add((noise(t),y,g))
rows=list(rows); random.shuffle(rows)
with open("data/comments.csv","w",newline="",encoding="utf-8") as f:
    w=csv.writer(f); w.writerow(["text","label","group"]); w.writerows(rows)
print("rows:",len(rows),"bullying:",sum(r[1] for r in rows))
