var player = new talkify.TtsPlayer().enableTextHighlighting();

   new talkify.playlist()
       .begin()
       .usingPlayer(player)
       .withTextInteraction()
       .withElements(document.querySelectorAll()) //<--Any element you'd like. Leave blank to let Talkify make a good guess
       .build() //<-- Returns an instance.
       .play();
