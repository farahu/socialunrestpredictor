import tweepy

auth = tweepy.OAuthHandler("K2iuG7AoH9dkN8ECMuPskt4wE", "yq65IjSNVpCjPKZX6xbi2mmBw5AzbtwlT38ekthOHSmUgZVkNf")
auth.set_access_token("2395628586-8VBCyvGsjmqnnNABZYw6ykURcZyVTfcJM33Kink", "000OmiUCdCjyTi7BUdn14TPW7SHMOGEEUMEUTxMuInV6F");

api = tweepy.API(auth)

public_tweets = api.home_timeline()
for tweet in public_tweets:
    print tweet.text




