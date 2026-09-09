# Phase 2 Intent Discovery Report: AmazonHelp

## 1. AmazonHelp Dataset Statistics

- **Total AmazonHelp-related tweets**: 109587
- **Support tweets**: 48993
- **Customer tweets**: 60594
- **Conversations reconstructed**: 23055

## 2. Conversation Reconstruction Methodology

1. **Filtered tweets**: Support tweets from `author_id == "AmazonHelp"` + `inbound == False`, and customer tweets mentioning `@AmazonHelp` or replying to AmazonHelp tweets.
2. **Reply graph**: Built using `in_response_to_tweet_id` → `tweet_id` edges.
3. **Conversation roots**: Customer tweets (inbound=True) with no parent in the filtered dataset.
4. **Thread reconstruction**: BFS from each root following reply edges, sorted chronologically.
5. **Minimum length**: Only conversations with ≥2 messages retained.

## 3. Conversation Statistics

| Metric | Value |
|--------|-------|
| Total Conversations | 23055 |
| Total Messages | 109587 |
| Customer Messages | 60594 |
| Support Messages | 48993 |
| Avg Messages/Conversation | 4.75 |
| Median Messages/Conversation | 4.0 |
| P90 Conversation Length | 9.0 |
| % With Support Reply | 99.93% |
| % Multi-Customer | 62.36% |
| % Multi-Support | 53.32% |

## 4. Resolution Heuristic

A conversation is marked as `resolved_candidate` if:
- Contains ≥1 customer message AND ≥1 support message
- AND (support message contains resolution keywords OR conversation has ≥3 messages)

**Resolution keywords**: refund, replacement, return, resolved, fixed, working, thank, dm, direct message, order, tracking, shipped, delivered, account, password, reset, login, issue resolved

**Note**: This is a heuristic, not ground truth. Manual verification needed.

## 5. Response Patterns (Top 15)

- **instructional**: 20614 (42.08%)
- **acknowledgment**: 14450 (29.49%)
- **informational**: 11995 (24.48%)
- **order_details**: 8058 (16.45%)
- **account_details**: 6171 (12.6%)
- **tracking_request**: 6098 (12.45%)
- **escalation**: 5301 (10.82%)
- **gratitude**: 3933 (8.03%)
- **dm_request**: 1334 (2.72%)
- **return_mention**: 1047 (2.14%)
- **resolution_confirm**: 936 (1.91%)
- **refund_mention**: 897 (1.83%)
- **replacement_mention**: 444 (0.91%)

## 6. Candidate Intent Clusters (KMeans, n=20)

| Cluster | Count | Top Terms | Example |
|---------|-------|-----------|---------|
| 0 | 63 | update, team, parcel, morning, getting, guys, order, need, kindle, waiting parcel | @AmazonHelp Sorry, I'm in Canada. I have checked the tracking (both on Amazon.ca and the company) an |
| 1 | 318 | amazon, amazon logistics, logistics, india, amazon india, item, sold, carrier, amazon pay, pay | @AmazonHelp Uno si es por amazon, otro bo |
| 2 | 131 | thanks, thanks help, help, just, link, ok, hi, issue, helpful, sent | @AmazonHelp Just received it !! Sorry and thanks 🙈 |
| 3 | 187 | help, need help, need, hi, don, hey, amazon, pls, card, actually | @AmazonHelp Who was helping me when we got disconnected and I have had little help from call 2 or 3  |
| 4 | 206 | delivered, today, delivered today, package, says, says delivered, order, items, package delivered, order delivered | @AmazonHelp And I paid extra to get it delivered quickly |
| 5 | 196 | prime, membership, prime membership, shipping, days, day, pay prime, prime member, pay, member | @AmazonHelp I've been charged full price membership for prime and not the student price, how do I go |
| 6 | 149 | account, amazon account, amazon, email, bank, locked, bank account, password, prime, ve | @AmazonHelp @115830 can you please tell me why money has been taken from my account when I haven’t o |
| 7 | 133 | que, la, el, por, lo, en, mi, si, pero, una | @AmazonHelp Ingresé a la página, introduje mi correo electrónico, solicite que me enviaran el código |
| 8 | 123 | did, reply, yes did, yes, time, delivery, amp, mail, told, waiting | @AmazonHelp So yesterday, you told me to contact you if my order hadn’t arrived today, I did so and  |
| 9 | 73 | amazon prime, prime, amazon, day, delivery, hi, prime account, paying, prime day, video | @AmazonHelp Then you should remove these Join Now options (in below images) from Amazon Prime app so |
| 10 | 1799 | today, time, ve, dm, ordered, days, don, says, yes, ich | @AmazonHelp its not good |
| 11 | 150 | customer, service, customer service, care, customer care, worst, amazon, help, called, customer support | @AmazonHelp And customer service says there is no way to contact. Need the order, not a refund. You  |
| 12 | 301 | just, refund, want, know, don, ve, don know, package, replacement, item | @AmazonHelp @115850 @115821 winner name please. Or just another fame quiz @144978 https://t.co/Isr5N |
| 13 | 40 | day requesting, requesting kind, kind response, requesting, kind, response, day, 19th, 13th, don time | @AmazonHelp @115830 @115851 423rd day requesting any kind of response |
| 14 | 334 | delivery, day, day delivery, date, prime, delivery date, today, guaranteed, time, says | @AmazonHelp my COD delivery is held by your delivery service |
| 15 | 82 | thank, resolved, ok, thank help, issue, sorted, ok thank, resolved thank, able, solved | @AmazonHelp Thank you!! for delivering the product. |
| 16 | 148 | email, sent, sent email, ve, received, got, saying, received email, email saying, confirmation email | @AmazonHelp No, no email updates to be found |
| 17 | 88 | app, amazon, kindle, alexa, amazon app, kindle app, mobile, iphone, music, issue | @AmazonHelp Don’t worry I have submitted delivery feedback via your app after item delivered late ye |
| 18 | 298 | order, order number, cancelled, number, cancel, refund, pre, pre order, received, cancel order | Hey @AmazonHelp!
Second delayed parcel in a row with Prime, and the best you can say is "sorry"?
Sho |
| 19 | 181 | est, le, pas, je, que, et, vous, ai, la, en | @AmazonHelp non ce n’est pas parfait. j’ai commandé hier pour être lové aujourd’hui et ça n’a pas ét |

## 7. Representative Examples by Cluster

### Cluster 0
- @AmazonHelp Sorry, I'm in Canada. I have checked the tracking (both on Amazon.ca and the company) and there has been no update since Monday evening when it arrived at the warehouse. Called courier and left message for callback, as well as emailed and no response.
- @AmazonHelp How is this not a simple fix? I preordered before beta keys existed, so just update the account so I can get the @116089 key
- @AmazonHelp I haven't received any update from your team, I have emailed you the details many times,  you guys have the worst service

### Cluster 1
- @AmazonHelp Uno si es por amazon, otro bo
- @AmazonHelp Do you think that I didn't tried that. I think local delivery boys promise means more than #amazon.
- @AmazonHelp From amazon

### Cluster 2
- @AmazonHelp Just received it !! Sorry and thanks 🙈
- Here we go again with the monkeys at Dynamex / @137088 ... Late deliveries and crappy parcel handling.  Thanks @AmazonHelp for continuing to screw your customers with the worst courier.
- @AmazonHelp Thanks. So, in summary, your tracking system is not to be relied upon for accuracy?

### Cluster 3
- @AmazonHelp Who was helping me when we got disconnected and I have had little help from call 2 or 3 other then a larger headache and a ugh to swear
- @AmazonHelp Please I need help, I do not know where I have my product.
- @AmazonHelp hi i need help with recharging a yearly prime sub and i accidently bought it when i was gonna buy this free month subscription and i really need help

### Cluster 4
- @AmazonHelp And I paid extra to get it delivered quickly
- @AmazonHelp Fudging when things are actually delivered isn't smart these days. Cameras everywhere. Where is the data?
- @AmazonHelp Prime proving to be useless AGAIN. Order should have arrived today by 9PM, and there now “may be a delay”. Waited in all day as you said it was going to be delivered. Sort it out.

### Cluster 5
- @AmazonHelp I've been charged full price membership for prime and not the student price, how do I go about verifying my student status?
- @AmazonHelp y'all charged me twice for my prime membership ...
- @AmazonHelp when u order something on prime on Thursday and still hasn’t arrived :/ if this is prime deffo won’t be signing up

### Cluster 6
- @AmazonHelp @115830 can you please tell me why money has been taken from my account when I haven’t ordered anything?? 😤🤔
- @AmazonHelp Even though my bank account is missing £79?! https://t.co/KLwR9vfeLv
- Can @AmazonHelp advise me on what a £7.99 charge on my account will be for?

### Cluster 7
- @AmazonHelp Ingresé a la página, introduje mi correo electrónico, solicite que me enviaran el código y aun no recibo nada. Necesito ingresar a mi cuenta.
- @AmazonHelp os dejado una consulta en MD, cuando podais por favor echadle un ojo, gracias por todo :)
- @AmazonHelp @132703 Además me mandan sms dándome 5 opciones, de las cuales, realmente solo me deja elegir la de recogida en agencia. Sinvergüenzas https://t.co/cDTWSArrR1

### Cluster 8
- @AmazonHelp So yesterday, you told me to contact you if my order hadn’t arrived today, I did so and you never replied. It seems like you’re ignoring me
- @12339 Fooling @115850 and defaming the name of @115821. Why in the earth would I request "Delay Delivery in customer's request"? @AmazonHelp  How did this happen without even contacting me? https://t.co/fjlix1w41y
- @AmazonHelp yes I did that.

### Cluster 9
- @AmazonHelp Then you should remove these Join Now options (in below images) from Amazon Prime app so no one will get confuse.. https://t.co/z6joqBOxOs
- @AmazonHelp @amazonhelp tomorrow? I thought the point of paying for amazon prime was so it arrived the next day?
- Amazon Prime video plays on my office internet and not home internet. How? What to do? @AmazonHelp

### Cluster 10
- @AmazonHelp its not good
- Dear @AmazonHelp  It feels  really devastating that once you promise something then you break it. I didn't get my cashback yet. See it. https://t.co/6mQg6QpQdc
- @AmazonHelp First date given 12/10, next 16/10 failed by ur courier partner,that's ur business partner.

### Cluster 11
- @AmazonHelp And customer service says there is no way to contact. Need the order, not a refund. You guys can’t track drivers? Never again.
- @AmazonHelp How do I request that none of my packages are shipped via USPS if this is the type of delivery service and “customer service” I receive?
- @AmazonHelp Don’t think your customer service is capable to be honest.

### Cluster 12
- @AmazonHelp @115850 @115821 winner name please. Or just another fame quiz @144978 https://t.co/Isr5NK9FvT
- @AmazonHelp Sadly not! Even the van men said there should be two parcels and they couldn’t find it! It’s quite an important parcel and don’t want it to be lost!
- @AmazonHelp  Hey , had ordered some stuffs Amazon IN ; had received damaged stuff ,despite raising 3 complaint seller refuse to entertain the replacement of stuff.
If the packaging is the issue why am not eligible for a refund. THis is not my fault just because of Supplier issue

### Cluster 13
- @AmazonHelp @115830 @115851 423rd day requesting any kind of response
- @AmazonHelp @115830 35th day requesting any kind of response
- @AmazonHelp @115830 132nd day requesting any kind of response

### Cluster 14
- @AmazonHelp my COD delivery is held by your delivery service
- @AmazonHelp But it shows me 10 days after delivery seriously! Like I'm a prime member yaar!and you says that it'll take maximum 1 or 2 day delivery time
- @122232 @AmazonHelp What’s going on? All day I’ve been in, nothing. Now my dog has to go hungry? https://t.co/mqNJ197ffL

### Cluster 15
- @AmazonHelp Thank you!! for delivering the product.
- @AmazonHelp Thank you for the response. I will look into it :')
- @AmazonHelp I'll Email. Thank you for your efforts. Have a nice day!

### Cluster 16
- @AmazonHelp No, no email updates to be found
- @AmazonHelp yes.  I'm not getting a verification code sent to my email.
- @AmazonHelp I can't talk. It has to be email etc.

### Cluster 17
- @AmazonHelp Don’t worry I have submitted delivery feedback via your app after item delivered late yesterday evening. Will see how @115830 react to my very negative feedback
- @AmazonHelp But at tme of retun window open when I try to return the mob 1 app installed and itself desided that there is no issue in mob @AmazonHelp
- @AmazonHelp 2. Die App und das ganze restliche Konstrukt taugt nicht zum untereinander zu kommunizieren.

### Cluster 18
- Hey @AmazonHelp!
Second delayed parcel in a row with Prime, and the best you can say is "sorry"?
Should I read it as "next time please order from sellers directly"?
- @AmazonHelp Just says "limited time" received yesterday, tried to order this morning.
- @AmazonHelp #unprofessional order id # 
403-7503264-7181114,403-8300003-6354735,403-5491238-8117144

### Cluster 19
- @AmazonHelp non ce n’est pas parfait. j’ai commandé hier pour être lové aujourd’hui et ça n’a pas été le cas. ce n’est pas parfait
- Ah et merci @AmazonHelp aussi de m’avoir envoyé un bouquin de collection dans une enveloppe à soufflet avec un livreur de merde 🤗 https://t.co/BETKxqnArs
- @AmazonHelp @408540 J’ai également le même problème, ma livraison a été décalé au 9 novembre alors que je suis membre Prime.


## 8. Data Quality Problems

1. **Missing reply links**: ~37% of tweets missing `in_response_to_tweet_id` - breaks conversation chains
2. **No explicit conversation_id**: Must infer from reply structure
3. **Timestamp granularity**: Second-level only, may misorder rapid exchanges
4. **Deleted/removed tweets**: Referenced tweets may not exist in dataset
5. **Bot/spam accounts**: Some "support" accounts may be automated

## 9. Potential Intent Taxonomy (Preliminary)

Based on clusters and response patterns:

1. **Delivery/Issue** - Package not arrived, late delivery, tracking
2. **Order Problem** - Wrong item, missing item, damaged, cancel order
3. **Refund/Return** - Request refund, return process, return label
4. **Account/Access** - Login issues, password reset, account locked
5. **Technical/App** - App not working, website error, Alexa/device issues
6. **Billing/Payment** - Unauthorized charge, billing question, payment failed
7. **Prime/Subscription** - Prime membership, subscription cancel, benefits
8. **Seller/Marketplace** - Third-party seller issues
9. **Gift Card** - Balance, redeem, not working
10. **General Inquiry** - Product questions, policy, how-to

## 10. Ambiguous/Boundary Cases

- **Delivery vs Order**: "Where is my order?" could be delivery tracking or order status
- **Refund vs Return**: Often coupled; customer wants money back AND to return item
- **Account vs Billing**: "Charge on my account" spans both
- **Technical vs General**: "App not working" could be bug or user error
- **Escalation needed**: Complex issues requiring specialist (fraud, legal, etc.)

## 11. Data Leakage Protection

**IMPORTANT**: Future evaluation must split at **conversation level**, not tweet level.
- No messages from same `conversation_id` in both training/retrieval and golden eval set
- Use `conversation_id` for stratified splits
- Golden set: 150-250 hand-labeled conversations (not individual tweets)

## 12. Next Steps

1. **Manual review** of candidate clusters to finalize intent taxonomy (10-15 intents)
2. **Create golden evaluation set** with conversation-level labels
3. **Build intent classifier** using resolved_candidate conversations as training data
4. **Build retrieval system** for historical resolution grounding
