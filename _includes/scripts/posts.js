(function () {
    var SOURCES = window.TEXT_VARIABLES.sources;

    function queryString() {
        var i = 0, queryObj = {}, pair;
        var queryStr = window.location.search.substring(1);
        var queryArr = queryStr.split('&');
        for (i = 0; i < queryArr.length; i++) {
            pair = queryArr[i].split('=');
            if (typeof queryObj[pair[0]] === 'undefined') {
                queryObj[pair[0]] = pair[1];
            } else if (typeof queryObj[pair[0]] === 'string') {
                queryObj[pair[0]] = [queryObj[pair[0]], pair[1]];
            } else {
                queryObj[pair[0]].push(pair[1]);
            }
        }
        return queryObj;
    }

    var setUrlQuery = (function () {
        var baseUrl = window.location.href.split('?')[0];
        return function (query) {
            if (typeof query === 'string') {
                window.history.replaceState(null, '', baseUrl + query);
            } else {
                window.history.replaceState(null, '', baseUrl);
            }
        };
    })();

    window.Lazyload.js(SOURCES.jquery, function () {
        var $tags = $('.js-tags');
        var $articleTags = $tags.find('button');
        var $tagShowAll = $tags.find('.tag-button--all');
        var $result = $('.js-result');
        var $articles = $result.find('.post-preview');
        var $lastFocusButton = null;
        var hasInit = false;

        function searchButtonsByTag(_tag) {
            if (!_tag) {
                return $tagShowAll;
            }
            var _buttons = $articleTags.filter('[data-encode="' + _tag + '"]');
            if (_buttons.length === 0) {
                return $tagShowAll;
            }
            return _buttons;
        }

        function buttonFocus(target) {
            if (target) {
                target.addClass('focus');
                $lastFocusButton && !$lastFocusButton.is(target) && $lastFocusButton.removeClass('focus');
                $lastFocusButton = target;
            }
        }

        function tagSelect(tag, target) {
            var _tag;
            $articles.each(function () {
                var $this = $(this);
                if (!tag || tag === '') {
                    $this.removeClass('d-none');
                } else {
                    var postTags = String($this.data('tags')).split(',');
                    if (postTags.indexOf(tag) > -1) {
                        $this.removeClass('d-none');
                    } else {
                        $this.addClass('d-none');
                    }
                }
            });
            if (!hasInit) {
                $result.removeClass('d-none');
                hasInit = true;
            }
            if (target) {
                buttonFocus(target);
                _tag = target.attr('data-encode');
                if (_tag === '' || typeof _tag !== 'string') {
                    setUrlQuery();
                } else {
                    setUrlQuery('?tag=' + _tag);
                }
            } else {
                buttonFocus(searchButtonsByTag(tag));
            }
        }

        var query = queryString(), _tag = query.tag;
        tagSelect(_tag);

        $tags.on('click', 'button', function () {
            tagSelect($(this).data('encode'), $(this));
        });
    });
})();
